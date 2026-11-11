import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

data ={
    'text':[
        "Congratulation! you've won a lottery. click hair to claim.",
        "Your invoice from our recent purchase is attached.",
        "Urgent: verify your account immediately!",
        "Hello: just checking in to see how you're doing.",
        "Important: update your payment information.",
        "Meet your new colleague at the office.",
    ],
    'label':[1,0,1,0,1,0]
}

df = pd.DataFrame(data)

print('Dataset: ')
print(df)

x = df['text']
y = df['label']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)
vectorizer = CountVectorizer()
x_train_vectorized = vectorizer.fit_transform(x_train)
x_test_vectorized = vectorizer.transform(x_test)

model = MultinomialNB()
model.fit(x_train_vectorized,y_train)

y_pred = model.predict(x_test_vectorized)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"\n Accuracy: {accuracy: .2f}")
print("Classification Report:")
print(report)
