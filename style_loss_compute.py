import numpy as np
import cv2
import os


# calculate the entire average rgb
def simple_calculate_avg_rgb(img):
    B_val = np.average(img[:,:,0])
    G_val = np.average(img[:,:,1])
    R_val = np.average(img[:,:,2])
    
    result_list = [B_val, G_val, R_val]
    result_arr = np.array(result_list)
    
    return result_arr

def simple_style_loss1(img1, img2):
    avg_img1 = simple_calculate_avg_rgb(img1)
    avg_img2 = simple_calculate_avg_rgb(img2)
    
    result = abs(avg_img1[0] - avg_img2[0])
    result = result + abs(avg_img1[1] - avg_img2[1])
    result = result + abs(avg_img1[2] - avg_img2[2])
    
    # average the three differences
    result = result/3
    
    # compare to max value 256 to regularize the result
    result = result/256
    
    return result


# calculate 9 parts of the image's average bgr 
def calculate_avg_rgb(img):
    rows_num = img.shape[0]
    cols_num = img.shape[1]
    
    row_step = rows_num//3
    col_step = cols_num//3
    
    result_list = []
    for row in range(3):
        next_row = row + 1
        for col in range(3):
            next_col = col + 1
            B_val = np.average(img[row*row_step:next_row*row_step, col*col_step:next_col*col_step, 0])
            G_val = np.average(img[row*row_step:next_row*row_step, col*col_step:next_col*col_step, 1])
            R_val = np.average(img[row*row_step:next_row*row_step, col*col_step:next_col*col_step, 2])
            # this list is generated from only 1/9 of the image
            a_list= [B_val, G_val, R_val]
            a_array = np.array(a_list)
            # append it to result_list
            result_list.append(a_array)
            
    result_arr = np.array(result_list)
    
    return result_arr



# simply calculate the difference of average rgb
def style_loss1(img1, img2):
    avg_9_img1 = calculate_avg_rgb(img1)
    avg_9_img2 = calculate_avg_rgb(img2)
    # the loss
    result = 0
    for i in range(9):
        intermediate_res = 0
        for j in range(3):
            diff = abs(avg_9_img1[i,j] - avg_9_img2[i,j])
            intermediate_res = intermediate_res + diff
        result = result + intermediate_res/3
    
    result = result/9
    
    result = result/256
    
    return result


# find the bound of given channel 0, 1, 2
def find_bound_with_channel(img, channel):
    
    rows_num = img.shape[0]
    cols_num = img.shape[1] 
    
    img_channel = img[:,:,channel]
    
    total_pixels = rows_num * cols_num
    
    count_up = 0
    # init upper bound
    upper_bound = 255
    
    while count_up/total_pixels < 0.2:
        count_up = np.sum(img_channel >= upper_bound)
        
        upper_bound = upper_bound-10
        
              
    # init lower bound
    lower_bound = 0
    count_low = 0
    
    while count_low/total_pixels < 0.2:
        count_low = np.sum(img_channel <= lower_bound)

        lower_bound = upper_bound+10
    
    return [lower_bound, upper_bound]
    
    
# img1 is style, img2 is transfered
def style_loss2(img1, img2):
    
    # get bounds of img1 and calculate the length 
    img1_B_bound = find_bound_with_channel(img1, 0)
    img1_B_length = img1_B_bound[1] - img1_B_bound[0]
    
    img1_G_bound = find_bound_with_channel(img1, 1)
    img1_G_length = img1_G_bound[1] - img1_G_bound[0]
    
    img1_R_bound = find_bound_with_channel(img1, 2)
    img1_R_length = img1_R_bound[1] - img1_R_bound[0]

    # get bounds of img2 and calculate the length 
    img2_B_bound = find_bound_with_channel(img2, 0)
    img2_B_length = img2_B_bound[1] - img2_B_bound[0]
    
    img2_G_bound = find_bound_with_channel(img2, 1)
    img2_G_length = img2_G_bound[1] - img2_G_bound[0]
    
    img2_R_bound = find_bound_with_channel(img2, 2)
    img2_R_length = img2_R_bound[1] - img2_R_bound[0]
    
    B_loss = abs(1 - img2_B_bound[0]/img1_B_bound[0]) + abs(1 - img2_B_bound[1]/img1_B_bound[1]) + abs(1 - img2_B_length/img1_B_length)
    
    # regularize the loss
    B_loss = B_loss/3
    
    G_loss = abs(1 - img2_G_bound[0]/img1_G_bound[0]) + abs(1 - img2_G_bound[1]/img1_G_bound[1]) + abs(1 - img2_G_length/img1_G_length)
    
    # regularize the loss
    G_loss = G_loss/3
    
    R_loss = abs(1 - img2_R_bound[0]/img1_R_bound[0]) + abs(1 - img2_R_bound[1]/img1_R_bound[1]) + abs(1 - img2_R_length/img1_R_length)
    
    # regularize the loss
    R_loss = R_loss/3    
    
    
    return [B_loss, G_loss, R_loss]


def style_loss3(img,style):
    img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    style_gray = cv2.cvtColor(style,cv2.COLOR_RGB2GRAY)
    ret,img_thresh = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    ret,style_thresh = cv2.threshold(style,127,255,cv2.THRESH_BINARY)
    img_perc=np.sum(img_thresh)/img_thresh.size
    style_perc=np.sum(style_thresh)/style_thresh.size
    return abs(img_perc/style_perc-1)

##################### testing evaluation ####################
#image_style = cv2.imread("./style1.jpg")
#image_transfered = cv2.imread("./stylized-image.jpg")

### the smaller loss, the better. 
#### average rgb difference to compute style loss, this is simpler than style_loss1
'''simple_style_loss1_res = simple_style_loss1(image_style, image_transfered)
print(simple_style_loss1_res)

###### average rgb difference to compute style loss #########
style_loss1_res = style_loss1(image_style, image_transfered)
print(style_loss1_res)


###### calculate the range of three channel and compute style loss #####
# it return [Blue_loss, Green_loss, Red_loss]
style_loss2_res = style_loss2(image_style, image_transfered)
print(style_loss2_res)'''
result_file=open("result.txt","r+")
style_g=os.listdir(r"g:\downloads\style")
img_g=os.listdir(r"g:\downloads\stylized_files")
for i in range(149):
    style_tmp=cv2.imread("style\\"+style_g[i])
    img_tmp=cv2.imread("stylized_files\\"+img_g[i])
    loss=style_loss3(img_tmp,style_tmp)
    result_file.write("file {0} loss: {1}\n".format(i,loss))
result_file.close()



