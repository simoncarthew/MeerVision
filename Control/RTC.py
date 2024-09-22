import smbus

class RTC:
    def __init__(self):
        self.bus = smbus.SMBus(1)  # I2C bus 1
        self.address = 0x68  # DS3231 I2C address

    def _bcd_to_dec(self, bcd):
        """ Convert binary-coded decimal (BCD) to decimal. """
        return (bcd & 0x0F) + ((bcd >> 4) * 10)

    def _dec_to_bcd(self, dec):
        """ Convert decimal to binary-coded decimal (BCD). """
        return (dec // 10) << 4 | (dec % 10)

    def read_time(self):
        """ Read current time from the RTC and return as a dictionary. """
        data = self.bus.read_i2c_block_data(self.address, 0x00, 7)

        seconds = self._bcd_to_dec(data[0])
        minutes = self._bcd_to_dec(data[1])
        hours = self._bcd_to_dec(data[2])
        day = self._bcd_to_dec(data[4])
        month = self._bcd_to_dec(data[5] & 0x1F)
        year = self._bcd_to_dec(data[6]) + 2000

        return {"year": year, "month": month, "day": day, "hours": hours, "minutes": minutes, "seconds": seconds}

    def set_time(self, year, month, day, hours, minutes, seconds):
        """ Set the RTC to the specified time. """
        self.bus.write_i2c_block_data(self.address, 0x00, [
            self._dec_to_bcd(seconds),
            self._dec_to_bcd(minutes),
            self._dec_to_bcd(hours),
            0,  # Day of the week (not used)
            self._dec_to_bcd(day),
            self._dec_to_bcd(month),
            self._dec_to_bcd(year - 2000)
        ])