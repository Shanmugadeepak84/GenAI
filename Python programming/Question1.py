#Get the number of subjects and marks for each subject from the user
No_of_subjects=input("Enter number of subjects")
print(No_of_subjects)
marks=[]
total_marks=0
print("Enter marks for each subject")

for i in range(int(No_of_subjects)):
    subject=input("Enter subject name")
    print(subject)
    subjectmarks=input(f"Enter marks for {subject} (out of 100):")
    print(f'{subject} marks is {subjectmarks}')
    marks.append(subjectmarks)
#Calculate the total marks and average score
for j in range(len(marks)):
    total_marks+=int(marks[j])

average_score=total_marks/int(No_of_subjects)
print("Total marks is:",total_marks)
print("Average score is:",average_score)

if(average_score>=90):
    print("Grade is A")
    print("Eligible for Honors")
elif(average_score>=80):
    print("Grade is B")
    print("Not eligible for Honors")
elif(average_score>=70):
    print("Grade is C")
    print("Not eligible for Honors")
elif(average_score>=60):
    print("Grade is D")
    print("Not eligible for Honors")
elif(average_score<50):
    print("Grade is F")
    print("Not eligible for Honors")




