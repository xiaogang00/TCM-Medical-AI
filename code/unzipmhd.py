import os

unzipfiles = [x for x in os.listdir('.') if x.find('.zip')!=-1]
print(unzipfiles)

for zipfile in unzipfiles:
  os.system('unzip -n '+ zipfile)
