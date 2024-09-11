## CAMERA TRAP

#### OUTPUTS
- [ ] Housing that's relatively robust to elements (Dust and Splash)
- [ ] Simple camera trap functionality
	- [ ] Take a single picture
	* [ ] Take photos for a set time or until stopped
* [ ] Process taken photos to produce raw data
	* [ ] {photo_id, time, number of meerkats, no_foraging, no_guarding, no_other, labeled_bounds_list}
* [ ] Automatically generated Plots
	* [ ] Meerkats No. vs Time

#### TESTING
- [ ] How long does pi last taking photos with and without solar panel
- [ ] How many photos can be taken

## PROCESSING

#### OUTPUTS
* [ ] Detect all meerkats in given frame
* [ ] Count the number of meerkats in a given frame
* [ ] Classify behaviour of individual meerkat

#### TESTING
* [ ] Compare the accuracy of varying OD models and obvious parameters
	* [ ] Accuracy of YOLO sizes and MEGA + CLASSIFIER
	* [ ] How does the seen data affect model performance
	* [ ] Compare Optimisers
* [ ] Compare the inference time of all the models on PI and PC