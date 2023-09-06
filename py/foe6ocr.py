import cv2
import numpy as np
import glob
import json
# import foe2json as m
# foe2json

class Saved:
  def __init__(self):
    self.digits = 0
    self.thr = 1
    # highest_threshold
    self.ht = []
  def set(self, digits, thr):
    self.digits = digits
    self.thr = thr
    self.ht.append(thr)
  def repl(self, digits, thr):
    self.digits = digits
    self.thr = thr
    self.ht[len(self.ht)-1] = thr

def group_digits_by_distance(sorted_distance, sorted_digits, sorted_pts, sorted_thr):
  grouped_digits = []
  current_group = []
  saved = Saved()

  for i in range(len(sorted_digits)):
    if i == 0:
      dx = 7
      dy = 0
    else: 
      dx = abs(sorted_pts[i][0] - sorted_pts[i - 1][0])
      dy = abs(sorted_pts[i][1] - sorted_pts[i - 1][1])
    rpt = f'{sorted_digits[i]}, dx={dx:2d}, dy={dy:2d}, pt={sorted_pts[i]}, t={sorted_thr[i]:.3f}'
  
  # digits are 6x10 and 7px apart. There are 5 rows of goods and they are 27px apart (y=6,33,60,87,114)
  # if two digits are competing for the same position, the lowest threshold wins

    if i == 0 or dy < 10:
      print(f'{sorted_distance[i]:3d} a=', rpt) 
      if dx > 3:
        current_group.append(sorted_digits[i])
        saved.set(sorted_digits[i], sorted_thr[i])
      else:
        print(f'       {sorted_digits[i]}, {sorted_thr[i]:.3f} < {saved.digits}, {saved.thr:.3f}', "replace" if sorted_thr[i] < saved.thr else "")
        if sorted_thr[i] < saved.thr:
          current_group[len(current_group)-1] = sorted_digits[i]
          saved.repl(sorted_digits[i], sorted_thr[i])
    else:
      if current_group:
        print('current_group=',current_group,'\n')
        grouped_digits.append(current_group)
        current_group = []
      print(f'{sorted_distance[i]:3d} b=',rpt)
      current_group.append(sorted_digits[i])
      saved.set(sorted_digits[i], sorted_thr[i])

  if current_group:
    print('current_group~',current_group)
    grouped_digits.append(current_group)
  print('highest_threshold: ',max(saved.ht),'\n')

  return grouped_digits

def squares(coordinates):
  return [((coord[1]//27) * 120 + coord[0]) for coord in coordinates]

def flatn(mainlist):
  return [item for sublist in mainlist for item in sublist]

def trimdb(db):
  for i in range(len(db)):
    db[i] = flatn(db[i])
    while len(db[i]):
      if db[i][len(db[i])-1] == '':
        db[i].pop()
      else:
        break

def dark(num): # dark BG by sect: 0,1,2,3 => 0,0,1,1
  return 0 if (num // 2) % 2 == 0 else 1

# digit_coordinates = {
#   '0': [],
#   '1': [(10, 37), (16, 37)],
#   '2': [(17, 10), (9, 91), (17, 118)],
#   '3': [],
#   '4': [(17, 64), (24, 64), (23, 91)],
#   '5': [(16, 91)],
#   '6': [(24, 118)],
#   '7': [(24, 10)],
#   '8': [(23, 37)],
#   '9': []
# }

def analyze(digit_coordinates, digit_thresholds):
  dbg = 1

  values = [
    # squares(digit_coordinates_json[str(i)]) 
    squares(digit_coordinates[str(i)]) 
    for i in range(10)]
  if dbg:
    print('digit_distance')
    # print(values)
    for i in range(10):
      print(f'{i}: {values[i]}') 

  digit_coordinates_json = {key: [(int(x), int(y)) for x, y in values] for key, values in digit_coordinates.items()}
  # pts = [digit_coordinates_json[str(i)] for i in range(10)]
  pts = [digit_coordinates[str(i)] for i in range(10)]
  if dbg:
    print('digit_coordinates')
    # print(flatn(pts))
    for i in range(10):
      print(f'{i}: {digit_coordinates[str(i)]}') 
  
  digit_thresholds_json = {key: [float(value) for value in values] for key, values in digit_thresholds.items()}
  # thr = [digit_thresholds_json[str(i)] for i in range(10)]
  thr = [digit_thresholds[str(i)] for i in range(10)]
  if dbg:
    print('digit_thresholds')
    # print(flatn(thr))
    for i in range(10):
      print(f'{i}: {digit_thresholds[str(i)]}') 

  digits = [[i] * len(values[i]) for i in range(10)]

  sorted_combined = sorted(zip( flatn(values), flatn(digits), flatn(pts), flatn(thr) ), key=lambda x: x[0])
  sorted_distance, sorted_digits, sorted_pts, sorted_thr = zip(*sorted_combined)
  if dbg:
    print('digit_sorted_combined')
    for i in range(len(sorted_combined)):
      print(f'{i}: {sorted_combined[i]}')
    print()

  grouped_digits = group_digits_by_distance(sorted_distance, sorted_digits, sorted_pts, sorted_thr)
  if dbg: print(grouped_digits)

  # [3, 3, 8] => 338  [int(), int(), int()] => str()
  results = [int(''.join(map(str, group))) for group in grouped_digits]
  if dbg: print(results)
  return results

# foe2ocr

def show(crop_image):
  cv2.imshow('crop',crop_image)
  k = cv2.waitKey(0) & 0xFF
  if k == 27:
    exit()
  cv2.destroyWindow('crop')

def template_matching(crop_image, templates, digit_coordinates, digit_thresholds):
    # Perform template matching for each digit
    for i, template in enumerate(templates):

      result = cv2.matchTemplate(crop_image, template, cv2.TM_SQDIFF_NORMED)
      threshold = 0.1  # seems like the sweetspot for this app (.031 - .081)
      loc = np.where(result < threshold)
      # print('i=',i,'loc=',loc)
      # show(crop_image)

      for pt in zip(*loc[::-1]):
        digit_coordinates[str(i)].append(pt)

      for j in range(len(loc[0])):
        threshold = result[loc[0][j]][loc[1][j]]
        digit_thresholds[str(i)].append(threshold)

    # print(digit_coordinates)
    # quit()
    result = analyze(digit_coordinates, digit_thresholds)
    return result

cropped_images_top = []
cropped_images_bottom = []
# Load the template images (individual digits)
templates = [cv2.imread(f'i/{i}.png', cv2.IMREAD_GRAYSCALE) for i in range(10)]
headings = [cv2.imread(f'i/h{i}.png', cv2.IMREAD_GRAYSCALE) for i in range(6)]
anchor = cv2.imread(f'i/anchor.png', cv2.IMREAD_GRAYSCALE)

cities = 4
eras = 12
db = [ [['','','','',''] for j in range(eras)] for i in range(cities) ]

for city in range(len(db)):
  # path = glob.glob('4/foe22c.png')
  path = glob.glob(f'{city+1}/*.png')
  print(f'\ncity={city}, path={path}')
  if not path: continue

  for png in path:
    large_image = cv2.imread(png, cv2.IMREAD_GRAYSCALE)

    # the anchor finds the inventory screen whether fullscreen or not

    result = cv2.matchTemplate(large_image, anchor, cv2.TM_SQDIFF_NORMED)
    mn,_,mnLoc,_ = cv2.minMaxLoc(result)
    MPx,MPy = mnLoc

    # the viewport is 69px below the anchor and 6px right, it's 690x373

    viewport = large_image[MPy+69:MPy+69+373,MPx+6:MPx+6+690]
    print(f'{mn:.3f}',mnLoc,'anchor') # ,viewport.shape,viewport.dtype)
    # show(viewport)


    for i, era in enumerate(headings):

      # headings are 690x26 and 167px apart, vertically
      # in the initial state, the 1st heading is 3px below the viewport
      # and there is 10px after the 3rd heading in the viewport

      result = cv2.matchTemplate(viewport, era, cv2.TM_SQDIFF_NORMED)
      mn,_,mnLoc,_ = cv2.minMaxLoc(result)
      MPx,MPy = mnLoc
      print(i,f'{mn:.3f}',mnLoc,f'heading[{i}]')
      if mn < .01 and MPy < 207: 

        # a full heading match is < .001
        # the cropbox is 32px below and 270px right of the heading
        # there are two of them 350px apart. They are 60x130
        # if the heading is > 207px down
        # then some data will be below the viewport

        for j in [0,1]:
          digit_coordinates = {str(i): [] for i in range(10)}
          digit_thresholds  = {str(i): [] for i in range(10)}

          crop_image = viewport[MPy+32:MPy+32+130,MPx+270+j*350:MPx+270+j*350+60]
          db[city][i*2+j] = template_matching(crop_image, templates, digit_coordinates, digit_thresholds)
          # show(crop_image)

          if j == 0: # dark(i*2+j): #
            cropped_images_top.append(crop_image)
          else:
            cropped_images_bottom.append(crop_image)

# print(json.dumps(db))
trimdb(db)
with open("D:/Users/shirha/Google Drive/foe_inventory.json", "w") as f:
  f.write(json.dumps(db))
for i in range(len(db)):
  print(f'{i}:{db[i]}')

top_row = np.hstack(cropped_images_top)
bottom_row = np.hstack(cropped_images_bottom)
complete_image = np.vstack((top_row, bottom_row))
cv2.imwrite("D:/Users/shirha/Google Drive/foe_inventory.png", complete_image)

'''
fullscreen
(base) D:/Downloads/Forge of Empires/chat_ocr>python foe5ocr.py
0 0.000 (602, 372) anchor
1 0.000 (608, 444)
2 0.000 (608, 611)
3 0.001 (608, 778)
4 0.339 (611, 778)
5 0.336 (608, 444)

windowed
(base) D:/Downloads/Forge of Empires/chat_ocr>python foe5ocr.py
0 0.000 (602, 391) anchor
1 0.000 (608, 463)
2 0.000 (608, 630)
3 0.001 (608, 797)
4 0.339 (611, 797)
5 0.336 (608, 463)

windowed page2 heading tolerance should be < .01
(base) D:/Downloads/Forge of Empires/chat_ocr>python foe5ocr.py
0 0.000 (602, 391) anchor
1 0.335 (608, 795)
2 0.272 (606, 461)
3 0.001 (608, 461)
4 0.001 (608, 628)
5 0.001 (608, 795)

windowed page2 scrolled slightly, notice tolerance index 5
(base) D:/Downloads/Forge of Empires/chat_ocr>python foe5ocr.py
0 0.000 (602, 391) anchor
1 0.328 (621, 481)
2 0.273 (606, 481)
3 0.001 (608, 481)
4 0.001 (608, 648)
5 0.063 (608, 815)
(base) D:/Downloads/Forge of Empires/chat_ocr>
'''
