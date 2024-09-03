import os
import requests
from pygbif import occurrences

# Define the species name and the directory to save images
species_name = "Suricata Desmarest"
output_dir = "Data/gbif_images"
os.makedirs(output_dir, exist_ok=True)

# Initialize pagination parameters
limit = 100
offset = 0
image_urls = []
searches = 1

while True:
    print("Search",searches)

    # Search for occurrences of the species in GBIF with pagination
    search_results = occurrences.search(scientificName=species_name, limit=limit, offset=offset)

    # Check if there are results; if not, break the loop
    if not search_results['results']:
        break

    # Extract image URLs from the occurrences
    for result in search_results['results']:
        if 'media' in result:
            for media in result['media']:
                # Ensure 'type' key exists before accessing it
                if media.get('type') == 'StillImage' and 'identifier' in media:
                    image_urls.append(media['identifier'])

    # Increment the offset to get the next batch of results
    offset += limit
    searches += 1

print("Found " + str(len(image_urls)))

downloaded = 690

# Download the images
for idx, url in enumerate(image_urls):
    if idx > 690:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                image_path = os.path.join(output_dir, f"{species_name.replace(' ', '_')}_{idx+1}.jpg")
                with open(image_path, "wb") as file:
                    file.write(response.content)
                print(f"Downloaded {image_path}")
            else:
                print(f"Failed to download image from {url}")
        except Exception as e:
            print(f"Error downloading image from {url}: {e}")

print(f"Downloaded {len(image_urls)} images for {species_name}.")
