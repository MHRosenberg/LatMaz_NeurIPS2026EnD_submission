#!/bin/bash

# Default directory
DIRECTORY="./"

# Parse command-line options
while getopts "d:" opt; do
  case $opt in
    d) DIRECTORY="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    exit 1
    ;;
  esac
done

echo "Checking directory: $DIRECTORY"

# Variable to track the currently displayed image
CURRENT_IMAGE=""

while true; do
    # Find the newest image
    NEWEST_IMAGE=$(find "$DIRECTORY" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.gif" \) -printf "%T@ %p\n" | sort -k 1nr | head -n 1 | cut -d' ' -f 2-)

    # Check if a new image is found
    if [ "$NEWEST_IMAGE" != "$CURRENT_IMAGE" ]; then
        echo "New image found: $NEWEST_IMAGE"

        # Kill the previous feh process, if running
        pkill feh
        
        # Update the current image
        CURRENT_IMAGE="$NEWEST_IMAGE"
        
        # Display the new image with feh
        #feh "$CURRENT_IMAGE" &
        #feh "$CURRENT_IMAGE" --auto-zoom &
        feh "$CURRENT_IMAGE" --geometry 800x400 &
        sleep 1  # Wait for feh to open
        wmctrl -r feh -e 0,0,0,1024,600  # Move feh to the top-left corner (small display)
    fi
    
    # Wait for 1 second before checking again
    sleep 1

done
