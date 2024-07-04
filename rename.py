import json

# Load the original JSON file
with open('C:\\Users\\shihr\\Desktop\\TMI2024\\moe_sam\\SAM-Med2D-800K\\image2label_train.json', 'r') as file:
    data = json.load(file)

# Create a new dictionary with the updated paths
new_data = {}
prefix = 'SAM-Med2D-800K/'

for key, value in data.items():
    new_key = prefix + key
    new_value = [prefix + v for v in value]
    new_data[new_key] = new_value

# Save the new JSON file
with open('C:\\Users\\shihr\\Desktop\\TMI2024\\moe_sam\\SAM-Med2D-800K\\image2label_train_updated.json', 'w') as file:
    json.dump(new_data, file, indent=4)

print("The file has been updated successfully.")
