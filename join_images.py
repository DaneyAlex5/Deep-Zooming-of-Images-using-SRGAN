import cv2
import numpy as np
def zoom():
  dataset = []
  def concat_tile(im_list_2d):
      return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])

  for i in range(1,10):
      for j in range(1,10):
          im=cv2.imread('images/I_0'+str(i)+'_0'+str(j)+'.png')
          dataset.append(np.array(im))
    
      im=cv2.imread('images/I_0'+str(i)+'_'+str(10)+'.png')
      dataset.append(np.array(im))
  
  for k in range(1,10):
    im=cv2.imread('images/I_'+str(10)+'_0'+str(k)+'.png')
    dataset.append(np.array(im))
  
  im=cv2.imread('images/I_'+str(10)+'_'+str(10)+'.png')
  dataset.append(np.array(im))
  im_zoomed = concat_tile([[dataset[0], dataset[1],dataset[2],dataset[3],dataset[4],dataset[5],dataset[6],dataset[7],dataset[8],dataset[9]],
                         [dataset[10], dataset[11],dataset[12],dataset[13],dataset[14],dataset[15],dataset[16],dataset[17],dataset[18],dataset[19]],
                         [dataset[20], dataset[21],dataset[22],dataset[23],dataset[24],dataset[25],dataset[26],dataset[27],dataset[28],dataset[29]],
                         [dataset[30], dataset[31],dataset[32],dataset[33],dataset[34],dataset[35],dataset[36],dataset[37],dataset[38],dataset[39]],
                         [dataset[40], dataset[41],dataset[42],dataset[43],dataset[44],dataset[45],dataset[46],dataset[47],dataset[48],dataset[49]],
                         [dataset[50], dataset[51],dataset[52],dataset[53],dataset[54],dataset[55],dataset[56],dataset[57],dataset[58],dataset[59]],
                         [dataset[60], dataset[61],dataset[62],dataset[63],dataset[64],dataset[65],dataset[66],dataset[67],dataset[68],dataset[69]],
                         [dataset[70], dataset[71],dataset[72],dataset[73],dataset[74],dataset[75],dataset[76],dataset[77],dataset[78],dataset[79]],
                         [dataset[80], dataset[81],dataset[82],dataset[83],dataset[84],dataset[85],dataset[86],dataset[87],dataset[88],dataset[89]],
                         [dataset[90], dataset[91],dataset[92],dataset[93],dataset[94],dataset[95],dataset[96],dataset[97],dataset[98],dataset[99]]])
  cv2.imwrite('Zoomed/Zoomed.png', im_zoomed)
