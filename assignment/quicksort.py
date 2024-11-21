import csv
# Writing to a file
data=[]

def quicksort(arr):
    if len(arr)<=1:
        return arr
    pivot = arr[-1]  # Choose the last element as the pivot
    less_than_pivot = []
    greater_than_pivot = []

    for item in arr[:-1]:  # Exclude the pivot
        if item[2] < pivot[2]:  # Compare by the third element (the score)
            less_than_pivot.append(item)
        else:
            greater_than_pivot.append(item)

    # Recursively apply quicksort and combine results
    return quicksort(less_than_pivot) + [pivot] + quicksort(greater_than_pivot)

with open('unsorted.txt','r',newline='') as csvfile:
    redaer=csv.reader(csvfile)
    for item in redaer:
        lastname,firstname,score=item
        data.append((lastname,firstname,int(score)))


sorted_data=quicksort(data)

# Appending to a file
with open('sorted.txt','w',newline="") as csvfile:
    writer=csv.writer(csvfile)
    for item in sorted_data:
        writer.writerow(item)
