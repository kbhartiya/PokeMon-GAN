import os
from PIL import Image

pokemon_dir = "./resizedData"
dstn = "./required_data"

for img in os.listdir(pokemon_dir):
	image = Image.open(os.path.join(pokemon_dir,img))
	print(image.mode)
	
	if image.mode == 'RGBA':
		image.load()
		background = Image.new('RGB',image.size,(0,0,0))
		background.paste(image,mask=image.split()[3])
		background.save(os.path.join(dstn,img.split('.')[0]+'.jpg'),"JPEG")
	
	if image.mode == 'P':
		#print(image.mode)
		image.convert('RGB')
		image.save(os.path.join(dstn,img.split('.')[0]+'.jpg'),"JPEG")
 
#print('Input data shape: {}'.format(orig_img.shape))
