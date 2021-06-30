from os import path 
import matplotlib.pyplot as plt 

while True :
  try :
    file_path = input("Enter the path of the file :  ")
    if file_path == 'exit' : exit(0)
    print("Path of the file : " , file_path )
    if path.isfile(file_path):
      fp = open(file_path,'r')
      #print(fp.read())
      lines , epoc_count , test_accu , val_accu = fp.readlines() , [] ,[] , [] 
      print("Info : " , lines[0])
      lines = lines[1:]
      for l in lines :
        if "Epoch" in l :
          epoc_count.append( int( l[ l.find(" ")+1 : l.find("/") ] ) ) 
          #print("Epoch" , epoc_count[-1] )
        else :
          acc , val_acc = l.find("accuracy: ") + 9  , l.rfind(":") + 1 
          test_accu.append( eval(l[acc : acc+7]) ) , val_accu.append( eval(l[val_acc : val_acc + 7 ]))
          #print("Accuracy : ", test_accu[-1] , "\nValidation Accuracy : " , val_accu[-1] ) 
      for i in range(len(epoc_count)) :
        print( epoc_count[i] , test_accu[i] , val_accu[i] )
      plt.xlabel("Epoch") , plt.ylabel("Accuracy")
      plt.plot( epoc_count , test_accu , 'r--' , label='Test accuracy' )
      plt.plot( epoc_count , val_accu  , 'b--' , label='Validation Accuracy' ) 
      plt.legend( loc = 'upper left' )
      plt.show() 
    else :
      print("File does not exist")
      continue 
  except Exception as e :
    print("Exception occure" , e )
    exit(0)