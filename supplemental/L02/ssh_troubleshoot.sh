# Troubleshooting ssh connection issues on macOS
$HOSTNAME_OR_IP="students.chapman.edu"  # Replace with the actual hostname or IP address

echo "1. Testing basic reachability..."
ping -c 3 $HOSTNAME_OR_IP

echo "2. Testing DNS resolution..."
dig $HOSTNAME_OR_IP

echo "3. Testing SSH over IPv4 only..."
ssh -4 -vvv $HOSTNAME_OR_IP

echo "4. Testing if port 22 is reachable..."
nc -vz $HOSTNAME_OR_IP 22

echo "5. Checking for firewall/stealth mode..."
/usr/libexec/ApplicationFirewall/socketfilterfw --getglobalstate
/usr/libexec/ApplicationFirewall/socketfilterfw --getstealthmode