import requests
import json


base_url = "https://ergast.com/api/f1"
driver_url = f"{base_url}/2023/drivers.json"
driver_response = requests.get(driver_url)
driver_data = driver_response.json()

driverToBeFound = "albon"

driver_list = driver_data['MRData']['DriverTable']['Drivers']

for driver in driver_list:
    if driverToBeFound == driver['driverId']:
        print(driver['nationality'])

else:
    print("Something is wrong")


