# Proximity-Palette-Swap
## Can you use the work in this project toward a project of your own?
You cannot sell the work or process in this project if you have not added additional value in some way.   
If you do add significant value, you are welcome to use this in your work, all I ask is credit this account please for the work I've done.  
(Basically just dont plagiarise me please :D )  
I would also love to see where this is used because I think its very fun and exciting!

## What is this project?
This algorithm was developed with the goal of taking in 2 pixel sprites, decomposing them into their core palettes and then applying one palette to the other. Initially inspired by the fusion mechanic in Siralim Ultimate, but with the pallete swapping I wish was in the game.

## What is the general process that this operates on?
The script takes the input of two sprites Steps 1 to 6 are identically completed on each sprite. Step 7 is where the program starts to consider them differently.

### Step 1: Generates a colour list
1.1. Searches through each pixel and creates a list of all colours in the source image, and how many of each colour occur in the image.
### Step 2: Generates a Spatial probability array: 
2.1. It also then counts how often a pixel of colour A appears next to colour B and adds all these numbers into a square array with the same length as the colour list. It may be important to note that its more specifically "if you chose a pixel of colour A, how many pixels of colour B will be directly adjacent", this means that the entry at [A,B] is not neccisarily the same as entry [B,A].  
2.2. It then divides these values by the frequency that colour A appears.
### Step 3: Generates a Colour Similarity Array
This program uses the CIELAB dE200 colour distance calculation, to determine how far colours are from each other.  
3.1. Converts an rgb value to XYZ colour space  
3.2. Converts the XYZ colour space to LAB colour space, (this conversion requires using a defined whitepoint and conversion matrix, I picked the first ones I saw but they all seem to have great variation)  
3.3. Then generated the distance between the two colours using the CIELAB dE200 formula.  
3.4. It then finds (1/(distance+1)) so that it returns 1 when the colours are exactly the same, and less than 1 when they are far from each other.
3.5. I dont think the program actually add this value into an array, but it may be easier to think of this way.  
### Step 4: Generates a final probability array
4.1. It takes a weighted average for each corresponding value in the two earier arrays, this allows the program to be adapted to each sprite set as required, since you can select to solely concern the spatial probability, or only the colour distance.
4.2. It removes all diagonal entries since all they will do is link a colour to itself which is unnecessary
4.3. Currently it also removes all links connected to "Special colours" which are currently pure black, purely transparent, and an old background colour I was using. This is so that the background and outlines will be handled differently. I would like to smarten this step in future.
### Step 5: Sorts the colours into groups based on the final probability array.
5.1.  It creates an array the length of the colour list, this is used to store the group number of each colour, initially all entries are zero.  
5.2.  It takes the maximum value of a line in the array (the program can work with either rows or columns) and puts this in a list of maxima.  
5.3. It checks to see if any maxima were zero, since this indicates that that colour is not next to anything other than black. This colour will get added to its own solo group.  
5.4. It takes the maximum from this list, and finds where this occurs in the probability array, if it occurs at the location [A,B] then A and B are given the same group index. If either colour is already in a group, the new colour gets the same group. If both colours are already in groups, then the two groups are merged. Lastly the maximum from the max list is deleted.  
5.5. The sorting of colours (step 5.4) continues until the list of group IDs no longer contains zero and all colours are assigned to groups.
### Step 6: The colour list is rearranged into a nested list of colours
6.1. The first level of a Palette array corresponds to the group, the second level contains each colour within that group.  
6.2. The first level is sorted by length, but the "Special colours" is also bumped to the front.   
6.3. The colours within each group are sorted based on their "human-lightness" which is their relative lightness that would be seen by a human eye.  
### Step 7: Palette stretching
This is where we have to consider Each sprite seperately, we have a figure-source, and a colour-source. The figure-source is the source of the shape of the final sprite, and can be thought of as the one that we paste the palette onto. The other is the one that we source the palette from and dictates the colours in the final image.   
7.1. The palette corresponding to the figure-source will remain unchanged. We must take the palette corresponding to the colour-source needs to be stretched to match the shape of the figure source.  
7.2. We take group A in the colour-source palette and stretch it to be the length of group A in the figure-source palette.   
7.2*. There is a special case for if the colour-source palette has an isolated colour, then we create our own 3 colour list, [screen blending mode with itself, original colour, multiply blending mode with itself] but this does not occur outside this special case. We then stretch the colour palette the same ay we do in 7.2.  
### Step 8: Apply the colour palette
8.1. We then take the original sprite, for each pixel in the figure-source sprite, we find its index in the figure-source palette array.
8.2. We set the corresponding pixel in the new image, to the corresponding colour in the colour-source palette array.

