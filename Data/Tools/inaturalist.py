import os
import requests
from pyinaturalist import get_observations

# Define the species name and the directory to save images
species_name = "Suricata suricatta"
output_dir = "Data/inaturalist_images"
os.makedirs(output_dir, exist_ok=True)

# Initialize variables
image_urls = set()  # Use a set to avoid duplicates
page = 1

while True:
    # Get observations for the species using the iNaturalist API
    observations = get_observations(q=species_name, per_page=100, page=page)
    
    if not observations['results']:
        break  # Exit loop if no more results

    # Extract image URLs from the observations
    for observation in observations['results']:
        if observation['photos']:
            for photo in observation['photos']:
                large_url = photo['url'].replace('square', 'large')  # Get higher resolution images
                image_urls.add(large_url)  # Add to set (avoids duplicates)
    
    print("Found " + str(len(image_urls)) + " unique images")

    page += 1  # Move to the next page

print("Found " + str(len(image_urls)) + " unique images")

# Download the images
for idx, url in enumerate(image_urls):
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
