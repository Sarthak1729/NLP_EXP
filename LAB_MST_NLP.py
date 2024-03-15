import nltk
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('maxent_ne_chunker')
nltk.download('words')

def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text.lower())

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [token for token in tokens if token not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    # Stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in lemmatized_tokens]

    return stemmed_tokens

def extract_entities(text):
    sentences = sent_tokenize(text)
    entities = []
    for sentence in sentences:
        tagged_words = pos_tag(word_tokenize(sentence))
        chunked = ne_chunk(tagged_words)
        for subtree in chunked:
            if isinstance(subtree, nltk.tree.Tree):
                entity = " ".join([word for word, tag in subtree.leaves()])
                entities.append(entity)
    return entities

def main():
    story = """ Yes. It’s 1986, I’m 9 years old and I come downstairs to get ready for school. My dad is reading the paper and he says, “Hey Corey, what do you know about comets?”

I immediately spout off everything a space-addicted 9 year old kid can/should know about comets. My dad beams at me and tells me bedtime will be postponed.

Night falls and my dad comes home early. We get into his Ford Escort and drive to K-Bee toys where he buys me a pair of GI Joe binoculars just as it is closing.

I can still remember what they looked like in the packaging.

We drive to a subdivision that is under construction and park on a mound of dirt that is now someone’s home.

We both lay on the hood of that car and look at the stars. We talk about space. We talk about life. He tells me that the next time this comet passes by Earth he will be long gone, but he hopes I will watch it and remember what it is like to be a child staring at the night sky with his dad.

We never saw Halley’s comet that night. Our binoculars were too cheap, the light pollution was too strong, the comet was too far away. To be honest, I think I prefer it that way.

Because what I want to remember about that day has nothing at all to do with a comet.

There’s a great many things I wanted for my life. Some of them have come to pass, others never will. But above all I hope that, one day, I will sit on a mound of dirt as an old man. I will sit with the great love of my life and our children and our children’s children.

And we, all of us, will look at the stars. And maybe we’ll see a comet and maybe we won’t.

But we’ll look regardless and we will think of the ones we have loved. """
    print("Original Story:")
    print(story)

    # Preprocess text
    preprocessed_text = preprocess_text(story)
    print("\nAfter Preprocessing (Tokenization, Stopword Removal, Lemmatization, and Stemming):")
    print(preprocessed_text)

    # Extract main entities
    entities = extract_entities(story)
    print("\nMain Entities in the Story:")
    print(entities)

if __name__ == "__main__":
    main()
