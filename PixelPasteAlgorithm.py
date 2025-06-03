#testing an algorithm for determining a creature's palette

##############################################################
#   PLEASE READ
##############################################################
# At the end of this script i have all the for loops that I've been using for testing 
# you'll need to edit the giant list of creature files, I have a bunch of individual
# sprites that I'm using to test.

# At the moment the code only spits out Creature B's palette applied onto Creature A
# And displays that image.

from PIL import Image,ImageFont, ImageDraw
import numpy as np
import random

##############################################################
#       DEFINTING FUNCTIONS
##############################################################

#This function takes in a color as a tuple and returns the lightness adjusted for the human eye 
def human_lightness(colour):
    lightness=np.sqrt( 0.299*(colour[0])**2 + 0.587*(colour[1])**2 + 0.114*(colour[2])**2 )
    return lightness

#This function converts rgb colour to (hue, chroma, lightness)
def rgb_to_hcl(r, g, b):
    # Normalize RGB values to the [0, 1] range
    r /= 255.0
    g /= 255.0
    b /= 255.0
    
    # Convert RGB to XYZ
    M=max(r, g, b)
    m=min(r, g, b)
    c=M-m

    #Calculate H
    if c == 0:
        Hprime=0
    elif M == r:
        Hprime=((g-b)/c)%6
    elif M == g:
        Hprime=((b-r)/c)+2
    elif M == b:
        Hprime=((r-g)/c)+4

    h=Hprime*60
    
    #l=0.5*(M+m) #basic lightness
    l=human_lightness((r,g,b)) #Human lightness
    return (h, c*255, l*255)

#This function is used in the conversion to XYZ colour
def sRGB_companding(V):
    V/=255
    if V <=0.4045:
        v=V/12.92
    else:
        v= ((V+0.055)/1.055)**2.4
    return v

#This function converts from rgb colour to XYZ colour
def rgb_to_xyz(R,G,B):
    '''gamma=2.2
    # Normalize RGB values to the [0, 1] range
    r=(R/255)**(gamma)
    g=(G/255)**(gamma)
    b=(B/255)**(gamma)'''
    

    r=sRGB_companding(R)
    g=sRGB_companding(G)
    b=sRGB_companding(B)

    #This guy has a list of matrices used by different companies
    M=[
        [0.6326696, 0.2045558, 0.1269946],
        [0.2284569, 0.7373523, 0.0341908],
        [0.0000000, 0.0095142, 0.8156958]
        ]
    
    X= M[0][0]*r + M[0][1]*g + M[0][2]*b
    Y= M[1][0]*r + M[1][1]*g + M[1][2]*b
    Z= M[2][0]*r + M[2][1]*g + M[2][2]*b
    return (X,Y,Z)

#This function is used in the conversion to LAB colours
def CIELAB_f(t):
    epsilon=0.008856
    k=903.3
    if t > epsilon:
        f=t**(1/3)
        return f
    else:
        f=((k*t)+16)/116
        return f

#This function converts to lab colours
def rgb_to_lab(r, g, b):
    XYZ = rgb_to_xyz(r, g, b)
    X=XYZ[0]*100
    Y=XYZ[1]*100
    Z=XYZ[2]*100

    
    L_star=116*CIELAB_f(Y/100)-16
    a_star=500*(CIELAB_f(X/96.4212)-CIELAB_f(Y/100))
    b_star=200*(CIELAB_f(Y/100)-CIELAB_f(Z/82.5188))
    return (L_star,a_star,b_star)

#This function returns the CIEdE00 distance between two colours
def dist_dE00(col1,col2):
    if col1==col2:
        return 0
    lab1=rgb_to_lab(col1[0], col1[1], col1[2])
    lab2=rgb_to_lab(col2[0], col2[1], col2[2])
    
    L_average_prime=(lab1[0]+lab2[0])/2
    
    C_1=( (lab1[1]**2) + (lab1[2]**2) )**(1/2)
    C_2=( (lab2[1]**2) + (lab2[2]**2) )**(1/2)
    C_average = (C_1+C_2)/2

    G=(0.5)*(1-(((C_average**7)/((C_average**7)+(25**7)))**0.5))
    
    a_1_prime = lab1[1]*(1+G)
    a_2_prime = lab2[1]*(1+G)

    
    
    C_1_prime=( (a_1_prime**2) + (lab1[2]**2) )**(1/2)
    C_2_prime=( (a_2_prime**2) + (lab2[2]**2) )**(1/2)
    C_average_prime=(C_1_prime+C_2_prime)/2
    
    h_1_prime=np.degrees(np.arctan2(lab1[2],a_1_prime))
    while h_1_prime < 0:
        h_1_prime+=360
    h_2_prime=np.degrees(np.arctan2(lab2[2],a_2_prime))
    while h_2_prime < 0:
        h_2_prime+=360
    
    if abs(h_1_prime-h_2_prime)>180:
        H_prime=(h_1_prime+h_2_prime+360)/2
    else:
        H_prime=(h_1_prime+h_2_prime)/2
    
    T=(1
       -0.17*np.cos(np.radians(H_prime-30))
       +0.24*np.cos(np.radians(2*H_prime))
       +0.32*np.cos(np.radians(3*H_prime+6))
       -0.20*np.cos(np.radians(4*H_prime-63))
       )
    
    if abs(h_2_prime-h_1_prime)<=180:
        Delta_h_prime=h_2_prime-h_1_prime
    elif abs(h_2_prime-h_1_prime)>180 and h_2_prime<= h_1_prime:
        Delta_h_prime=h_2_prime-h_1_prime+360
    else:
        Delta_h_prime=h_2_prime-h_1_prime-360

    Delta_L_prime=lab2[0]-lab1[0]

    Delta_C_prime=C_2_prime-C_1_prime

    Delta_H_prime=2*((C_2_prime*C_1_prime)**(0.5))*np.sin(np.radians(Delta_h_prime/2))
    
    S_L=1+(
        (0.015*((L_average_prime-50)**2))/
        (   (
            20+((L_average_prime-50)**2)
            )*(0.5))
        )
    S_C=1+0.045*C_average_prime
    S_H=1+0.015*C_average_prime*T

    Delta_theta=30*np.exp(
        -(( (H_prime-275)/(25) )**2)
        )
    R_C=2*(( (C_average_prime**7)/(C_average_prime**7+(25**7)) )**(0.5))
    R_T=-R_C*np.sin(np.radians(2*Delta_theta))

    k_L=10
    k_C=1
    k_H=1

    dist=np.sqrt(
        ((Delta_L_prime/(k_L*S_L))**2)+
        ((Delta_C_prime/(k_C*S_C))**2)+
        ((Delta_H_prime/(k_H*S_H))**2)+
        R_T*(
            ((Delta_C_prime/(k_C*S_C)))+
            ((Delta_H_prime/(k_H*S_H)))
            ))
    return dist

# This function takes in an image and returns a list of colours
# and a probability array.
# The columns align with the centre colour, and the rows represent the neighbour's colour

# The weighting variable determines how much you want to consider a colour's Cielab2000 distance
# in the calculation, a value of 0 gives no consideration to the colour distance
def colour_prob(im,weighting):
    pixels = im.load()
    width, height = im.size

    color_list = []
    count_dict = {}

    # Tally up each pixel colour into a dictionary
    for x in range(width):
        for y in range(height):
            color = pixels[x, y]
            if color not in count_dict:
                color_list.append(color)
                count_dict[color] = 1
            else:
                count_dict[color] += 1

    # Build neighbour count dictionary
    prob_dict = {}
    for x in range(width):
        for y in range(height):
            current_color = pixels[x, y]
            if current_color not in prob_dict:
                prob_dict[current_color] = {}

            # Check neighbors: up, down, left, right
            neighbors = []
            if y > 0: neighbors.append(pixels[x, y - 1])     # Up
            if y < height - 1: neighbors.append(pixels[x, y + 1])  # Down
            if x > 0: neighbors.append(pixels[x - 1, y])     # Left
            if x < width - 1: neighbors.append(pixels[x + 1, y])   # Right

            for neighbor in neighbors:
                if neighbor not in prob_dict[current_color]:
                    prob_dict[current_color][neighbor] = 1
                else:
                    prob_dict[current_color][neighbor] += 1

    # Build normalized proability matrix
    num_colors = len(color_list)
    col_array = np.zeros((num_colors, num_colors))

    for col1 in prob_dict:
        for col2 in prob_dict[col1]:

            #Chance that two colours are in the same palette because of distance:
            CIELAB_chance=1/(dist_dE00(col1,col2)+1)

            #Chance that two colours are in the same palette because of distance:
            normalized_chance = (prob_dict[col1][col2]/(4*count_dict[col1]))

            #Take the average to get the final probability
            final_Probability=(weighting*CIELAB_chance) + ((1-weighting)*normalized_chance)

            
            i = color_list.index(col1)
            j = color_list.index(col2)
            col_array[i, j] = final_Probability
    return color_list, col_array

# This function takes in the colour list and colour array, considers the maximum of each column
# then adds colours to their most likely neighbours until all colours are in a group
def colour_group(color_list,col_array):
    Special_colours=[(0, 0, 0, 255),(21, 21, 21, 255),(0, 0, 0, 0)]
    n_colors = len(color_list)

    removal_ids=[]
    for colour_i in Special_colours:
        if colour_i in color_list:
            idx_i = color_list.index(colour_i)
            removal_ids.append(idx_i)

    for x in range(n_colors):
        col_array[x][x] = 0
    
    for i in removal_ids:
        for x in range(n_colors):
            col_array[i][x] = 0
            col_array[x][i] = 0

    for i in removal_ids:
        for j in removal_ids:
            if i != j:
                col_array[i][j] = 5
                col_array[j][i] = 5
    
    group_id = [0] * n_colors
    # Find max value in each column
    max_list = [max(col_array[:, col]) for col in range(n_colors)]

    # Assign group numbers to isolated colors (with no connections)
    while 0 in max_list:
        idx = max_list.index(0)
        group_id[idx] = max(group_id) + 1
        max_list[idx] = -1  # mark as processed
    # Cluster colors by strongest connections
    while 0 in group_id:
        max_val = max(max_list)
        max_list[max_list.index(max_val)] = 0  # mark this one as used

        # Get all (in_col, out_col) pairs with this max similarity
        indices = np.argwhere(col_array == max_val)
        for in_col, out_col in indices:
            group_in = group_id[in_col]
            group_out = group_id[out_col]

            if group_in == 0 and group_out == 0:
                # Both ungrouped — assign a new group
                new_group = max(group_id) + 1
                group_id[in_col] = group_id[out_col] = new_group
            elif group_in == 0:
                group_id[in_col] = group_out
            elif group_out == 0:
                group_id[out_col] = group_in
            elif group_in != group_out:
                # Merge all g_out into g_in
                for i in range(n_colors):
                    if group_id[i] == group_out:
                        group_id[i] = group_in
    #Reduce to minimum number of integers
    return group_id

# This function takes in the colour list and group IDs and sorts all colours in a group
# into runs that are sorted by lightness
def palettesort_2D(color_list, group_ids):
    
    grouped_colors = [[] for _ in range(max(group_ids))]
    for idx, group in enumerate(group_ids):
        grouped_colors[group - 1].append(color_list[idx])
    for group in grouped_colors:
        group.sort(key=human_lightness)
    grouped_colors.sort(key=len, reverse=True)
    
    #remove black series at that position and place it at the front
    for i in grouped_colors:
        if (0,0,0,0) in i or (0,0,0,255) in i or (21,21,21,255) in i:
            OriginalList=i
            black_and_bg_idx = grouped_colors.index(i)
    grouped_colors.pop(black_and_bg_idx)
    grouped_colors.insert(0,OriginalList)
    #remove all empty palettes
    while [] in grouped_colors:
        grouped_colors.pop(grouped_colors.index([]))
        
    Palettecheck=Image.new(mode="RGBA",size=(len(grouped_colors),len(color_list)),color=(0,0,0,0))
    pixels=Palettecheck.load()
    for i in range(len(grouped_colors)):
        for j in range(len(grouped_colors[i])):
            pixels[i,j]=grouped_colors[i][j]
    #Palettecheck.show()
    #print(grouped_colors)
    return grouped_colors
    
# This function controls the stretching or squishing of one gradient to match the length of another
# Both palette_image and palette_apply are only one of the colour runs so they are only a single list of colours.
def runA_to_runB(palette_image,palette_apply):
    
    RequiredLength=len(palette_image)
    CurrentLength=len(palette_apply)

    if RequiredLength==CurrentLength:
        return palette_apply
    else:
        Required_indices=np.linspace(0, 1, RequiredLength)
        Current_indices = np.linspace(0, 1, CurrentLength)
        
        rgbaRestruct=[[],[],[],[]]
        for i in range(len(Current_indices)):
            for rgba in range(4):
                rgbaRestruct[rgba].append(palette_apply[i][rgba])

        newpalette_rgba=[[],[],[],[]]   
        for i in range(4):
            newpalette_rgba[i] = np.interp(Required_indices,Current_indices,rgbaRestruct[i])
        newpalette=[]
        for i in range(len(newpalette_rgba[0])):
            colour=(int(round(newpalette_rgba[0][i],0)),
                    int(round(newpalette_rgba[1][i],0)),
                    int(round(newpalette_rgba[2][i],0)),
                    int(round(newpalette_rgba[3][i],0)))
            newpalette.append(colour)
        
    return newpalette

# This function just replaces all colours in an image to the new palette
# Also the pixels method does weird object stuff so in order to not edit the original source images,
# I needed to make a new image imnew
def pixel_by_pixel(im,pA,pB):
    imnew=Image.new(mode="RGBA",size=im.size,color=(0,0,0,0))
    pixelsold = im.load()
    pixelsnew = imnew.load()
    for i in range(im.size[0]): # for every pixel:
        for j in range(im.size[1]):
            if pixelsold[i,j]in pA:
                Index=pA.index(pixelsold[i,j])
                pixelsnew[i,j]=pB[Index]
            else:
                ''
    return imnew
    
# This function just wraps up the hole process of applying each function because I wanted to test out a whole bunch of source creatures
def PalettePaste(Creature_in,Creature_out,Adist,Bdist,offset):
    # Get the Palette for Creature_in 
    ColorList_in,ColorArray_in=colour_prob(Creature_in,Adist)
    GroupList_in=colour_group(ColorList_in,np.transpose(ColorArray_in))
    PaletteArray_in=palettesort_2D(ColorList_in,GroupList_in)

    # Get the Palette for Creature_out
    ColorList_out,ColorArray_out=colour_prob(Creature_out,Bdist)
    GroupList_out=colour_group(ColorList_out,np.transpose(ColorArray_out))
    PaletteArray_out=palettesort_2D(ColorList_out,GroupList_out)

    # Both Palette arrays are lists of colour ranges, uncomment the .show() at the end of
    # palettesort_2D if you'd like to see what that looks like
    
    # This whole chunk takes in a list of colours and stretches them or
    # squishes them to fit the destination
    
    Pal_og=[]
    Pal_in=[]
    ApplyLen=len(PaletteArray_in)
    
    for i in range(ApplyLen):
        if i == 0:
            "" #Leave the first entries alone because black and background shouldn't change
        else:
            #This allows us to keep cycling through colour runs if one creature
            # has 3 groups and we want to apply that to a creature with 7 groups
            b_idx=1+((i+offset)%(len(PaletteArray_out)-1))
                
            #if the run is larger than 2, do the regular procedure
            if len(PaletteArray_in[i])>2:
                NewPaletteApply=runA_to_runB(PaletteArray_in[i],PaletteArray_out[b_idx])
                for j in range(len(NewPaletteApply)):
                    Pal_og.append(PaletteArray_in[i][j])
                    Pal_in.append(NewPaletteApply[j])

            #If it is exactly two copy over the first and the last valused from the PaletteArray_out
            elif len(PaletteArray_in[i])==2:
                Pal_og.append(PaletteArray_in[i][0])
                Pal_in.append(PaletteArray_out[b_idx][0])
                Pal_og.append(PaletteArray_in[i][1])
                Pal_in.append(PaletteArray_out[b_idx][-1])
            
            #If it is exactly one copy over the first colours only
            elif len(PaletteArray_in[i])==1:
                
                Pal_og.append(PaletteArray_in[i][0])
                Pal_in.append(PaletteArray_out[b_idx][0])
            else:
                print("oh no")

    #This function replaces each pixel with colour in Pal_og, with the corresponding colour in Pal_in
    Pasted_image=pixel_by_pixel(Creature_in,Pal_og,Pal_in)

    return Pasted_image






def distance_verify(Colourlist):
    #Create an image to check the result
    Col_dist_im=Image.new(mode="RGBA",size=((len(Colourlist)+1)**2,3),color=(0,0,0,0))
    pixels=Col_dist_im.load()
    for i in range(len(Colourlist)):
        for j in range(len(Colourlist)):
            pixels[i*len(Colourlist)+j,0] = Colourlist[i]
            pixels[i*len(Colourlist)+j,1] = Colourlist[j]
            
            CIELAB_distance=dist_dE00(Colourlist[i],Colourlist[j])
            if CIELAB_distance>40:
                pixels[i*len(Colourlist)+j,2]=(255,0,0,255)#RED
            elif CIELAB_distance<=40 and CIELAB_distance>35:
                pixels[i*len(Colourlist)+j,2]=(255,128,0,255)#ORANGE
            elif CIELAB_distance<=35 and CIELAB_distance>30:
                pixels[i*len(Colourlist)+j,2]=(255,255,0,255)#yellow
            elif CIELAB_distance<=30 and CIELAB_distance>25:
                pixels[i*len(Colourlist)+j,2]=(0,255,0,255)#green
            elif CIELAB_distance<=25 and CIELAB_distance>20:
                pixels[i*len(Colourlist)+j,2]=(0,128,255,255)#blue
            elif CIELAB_distance<=20:
                pixels[i*len(Colourlist)+j,2]=(255,0,255,255)#magenta
            else:
                pixels[i*len(Colourlist)+j,2]=(0,255,0,255)
    Big_for_show=Col_dist_im.resize((10*Col_dist_im.size[0],10*Col_dist_im.size[1]),resample=0)
    Big_for_show.show()

##############################################################
#   OPENING ALL TEST IMAGES
##############################################################

# open method used to open different extension image file
Creature_test = Image.open("./Creature Placeholder test.png")
Creature_A = Image.open("./Creature Placeholder A.png")
Creature_B = Image.open("./Creature Placeholder B.png")
Creature_C = Image.open("./Creature Placeholder C.png")
Creature_D = Image.open("./Creature Placeholder D.png")
Creature_E = Image.open("./Creature Placeholder E.png")
Creature_F = Image.open("./Creature Placeholder F.png")
Creature_G = Image.open("./Creature Placeholder G.png")
Creature_H = Image.open("./Creature Placeholder H.png")
Creature_I = Image.open("./Creature Placeholder I.png")
Creature_J = Image.open("./Creature Placeholder J.png")
Creature_K = Image.open("./Creature Placeholder K.png")
Creature_L = Image.open("./Creature Placeholder L.png")
Full_Texture = Image.open("./Whole big texture.png")
#I add all images to a list just so I can cycle through them all in the for loops below
Creature_List=[Creature_test,
               Creature_A,  #fox
               Creature_B,  #apocalypse
               Creature_C,  #Snake chef
               Creature_D,  #fire dragon
               Creature_E,  #sabertooth tiger
               Creature_F,  #Wyvern
               Creature_G,  #Toxic sludge
               Creature_H,  #Golem
               Creature_I,  #cerberus
               Creature_J,  #pheasant
               Creature_K,  #Dragon
               Creature_L   #Celestial Eye
               ]

##############################################################
#   UNPACK THE BIG TEXTURE
##############################################################
Big_list = Image.open("./adjusted.png")
Tex_Start=196
CreatureCount=8
Imagelist=[]

for i in range(CreatureCount):
    x=random.randint(0,56)
    y=random.randint(0,2)
    NewimSprite=Big_list.crop(((x*68)+Tex_Start,68*y,Tex_Start+((x+1)*68),((y+1)*68)))
    Imagelist.append(NewimSprite.crop((2,2,66,66)))
#print(Imagelist)

##Eight_by_Eight=Image.new(mode="RGBA",size=(CreatureCount*68,CreatureCount*68),color=(0,0,0,0))
##
##for i in range(CreatureCount):
##    for j in range(CreatureCount):
##        Nthim=PalettePaste(Imagelist[i],Imagelist[j],0.2,0.2,0)
##        Image.Image.paste(Eight_by_Eight,Nthim,(64*i,64*j))
##
##Eight_by_Eight.show()

# Offset testing
Offset_distance=10
Offset_test_im=Image.new(mode="RGBA",size=(Offset_distance*68,2*68),color=(0,0,0,0))

Image.Image.paste(Offset_test_im,Imagelist[0],(0,0))
Image.Image.paste(Offset_test_im,Imagelist[1],(68,0))

for i in range(Offset_distance):
    Nthim=PalettePaste(Imagelist[0],Imagelist[1],0.2,0.2,i)
    Image.Image.paste(Offset_test_im,Nthim,(64*i,68))

Offset_test_im.show()
##############################################################
#   USING THE CODE
##############################################################

##imtest=PalettePaste(Creature_A,Creature_B,0.2,0.2)
##imtest.show()

#Test random creatures in the big texture


# Create new empty images to hold all our tests
NumberOfCreatures=len(Creature_List)
Eight_by_Eight=Image.new(mode="RGBA",size=(NumberOfCreatures*64,NumberOfCreatures*64+50),color=(0,0,0,0))

ThresholdTests=5
Threshold_testing=Image.new(mode="RGBA",size=(ThresholdTests*64,ThresholdTests*64),color=(0,0,0,0))



 # Testing what the effect of weighting is, k iterates through a few weighting amounts
##
##for k in [1]:
##    for i in range(NumberOfCreatures):
##        for j in range(NumberOfCreatures):
##            Nthim=PalettePaste(Creature_List[i],Creature_List[j],0.2*k,0.2*k)
##            Image.Image.paste(Eight_by_Eight,Nthim,(64*i,64*j))
##            
##    I1 = ImageDraw.Draw(Eight_by_Eight)
##    I1 .rectangle([0,NumberOfCreatures*64,(50*(k+1)),NumberOfCreatures*64+50], fill=(255,0,0,255))
##    Eight_by_Eight.show()
##
##for i in range(ThresholdTests):
##    for j in range(ThresholdTests):
##        Nthim=PalettePaste(Creature_J,Creature_J,(i)*0.2,(j)*0.2)#i==j)
##        Image.Image.paste(Threshold_testing,Nthim,(64*i,64*j))
##Threshold_testing.show()

