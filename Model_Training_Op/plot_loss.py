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
      lines , epoc_count , test_loss , val_loss = fp.readlines() , [] ,[] , [] 
      print("Info : " , lines[0])
      lines = lines[1:]
      for l in lines :
        if "Epoch" in l :
          epoc_count.append( int( l[ l.find(" ")+1 : l.find("/") ] ) ) 
          #print("Epoch" , epoc_count[-1] )
        else :
          t_loss , v_loss = l.find("loss: ") + 6  , l.rfind("val_loss:") + 10 
          test_loss.append( eval(l[ t_loss : t_loss+7]) ) , val_loss.append( eval(l[v_loss : v_loss + 7 ]))
          #print("Loss : ", l[acc : acc+7] , "\nValidation Losss : " , l[val_acc : val_acc + 7 ] ) 
      
      for i in range(len(epoc_count)) :
        print( epoc_count[i] , test_loss[i] , val_loss[i] )
      print(max(val_loss) , epoc_count[ val_loss.index( max(val_loss) ) ] )
      plt.xlabel("Epoch") , plt.ylabel("Accuracy")
      plt.plot( epoc_count , test_loss , 'r--' , label='Training Loss' )
      plt.plot( epoc_count , val_loss  , 'b--' , label='Validation Loss' ) 
      plt.legend( loc = 'upper left' )
      plt.show() 
    else :
      print("File does not exist")
      continue 
  except Exception as e :
    print("Exception occure" , e )
    exit(0)
