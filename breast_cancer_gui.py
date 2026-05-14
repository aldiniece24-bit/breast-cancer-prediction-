import numpy as np
from tkinter import *

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load dataset
data = load_breast_cancer()

X = data.data
y = data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
    
# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Prediction function
def predict_cancer():
    try:
        r = float(radius.get())
        t = float(texture.get())
        p = float(perimeter.get())
        a = float(area.get())
        s = float(smoothness.get())

        # Use dataset average values
        data_input = list(X.mean(axis=0))

        data_input[0] = r
        data_input[1] = t
        data_input[2] = p
        data_input[3] = a
        data_input[4] = s

        data_input = np.array(data_input).reshape(1,-1)

        # Scale input
        data_input = scaler.transform(data_input)

        result = model.predict(data_input)

        if result[0] == 0:
            output_label.config(
                text="⚠ Malignant (Breast Cancer Detected)",fg="red")
        else:
            output_label.config(
                text="✓ Benign (No Cancer)",fg="green")

    except ValueError:
        output_label.config(text="Please enter valid numbers", fg="orange")
    except Exception as e:
        output_label.config(text="Error: " + str(e), fg="orange")
        print(e)

# Clear button function
def clear_fields():
    radius.set("")
    texture.set("")
    perimeter.set("")
    area.set("")
    smoothness.set("")
    output_label.config(text="")


# GUI Window
root = Tk()
root.title("AI Breast Cancer Detection System")
root.geometry("400x450")

Label(root, text=" Breast Cancer Prediction", font=("Arial",16)).pack(pady=10)

# Input variables
radius = StringVar()
texture = StringVar()
perimeter = StringVar()
area = StringVar()
smoothness = StringVar()

Label(root,text="Mean Radius").pack()
Entry(root,textvariable=radius).pack()

Label(root,text="Mean Texture").pack()
Entry(root,textvariable=texture).pack()

Label(root,text="Mean Perimeter").pack()
Entry(root,textvariable=perimeter).pack()

Label(root,text="Mean Area").pack()
Entry(root,textvariable=area).pack()

Label(root,text="Mean Smoothness").pack()
Entry(root,textvariable=smoothness).pack()

Button(root,text="Predict",command=predict_cancer).pack(pady=10)
Button(root,text="Clear",command=clear_fields).pack()

output_label = Label(root,text="",font=("Arial",12))
output_label.pack(pady=10)
root.mainloop()