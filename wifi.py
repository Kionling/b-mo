import subprocess

def list_wifi_networks():
    # Path to the Airport utility on macOS
    airport_path = "/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport"
    try:
        # Scan for available networks
        result = subprocess.run([airport_path, "-s"], capture_output=True, text=True, check=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Failed to list WiFi networks. Error: {e}")


def connect_to_wifi():
    while True:
        ssid = input("Enter the SSID of the WiFi network or type 'list' to display all available networks. Type 'quit' to leave: ")
        if ssid.lower().strip() == 'list':
            list_wifi_networks()
            continue
        elif ssid.lower() == 'quit':
            print('Exiting...');
            exit()
            break

    password = input("Enter the WiFi password: ")

    try:
        # Delete existing connection with the same SSID (optional, to avoid conflicts)
        subprocess.run(["nmcli", "con", "delete", ssid], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Add a new WiFi connection
        subprocess.run(["nmcli", "dev", "wifi", "connect", ssid, "password", password], check=True)
        print(f"Successfully connected to {ssid}.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to connect to {ssid}. Error: {e}")

if __name__ == "__main__":
    connect_to_wifi()
