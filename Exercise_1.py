# Import necessary libraries
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from nltk.corpus import brown
from string import punctuation
from nltk.tokenize import word_tokenize
import re
from scipy.stats import linregress, ks_2samp

# True for saving, false for showing
save_plots = True

# Folder in which the plots will be saved
# It should end in a slash
# An empty string will save them in the current directory
plot_save_adress = ""

# Open the 10 files from the Gaelic corpus
# A for-loop is used to avoid having to type all the adresses
# The final adress (ns10) is typed because it's easier
# The files have the format "word/tag" separated by spaces
# We use split to get the word-pair tags and make a list
text_raw_gaelic = []
for i in range(9):
    t = open(f"corpus_ex1/ns0{i+1}.txt", encoding="utf-8").read().split()
    text_raw_gaelic += t
t = open(f"corpus_ex1/ns10.txt", encoding="utf-8").read().split()
text_raw_gaelic += t

# We split the word-tag pairs by the "/" character
# If they have a tag we check that it's not punctuation (tag X)
# If it's not we add it to a list
# We also add the words with no tag (they are actual words we checked)
tokens_gaelic = []
for i in text_raw_gaelic:
    t = i.split("/")
    if len(t) == 2 and not t[1].lower().startswith("x"):
        tokens_gaelic.append(t[0].lower())
    elif len(t) != 2:
        tokens_gaelic.append(t[0].lower())

# Here we use the news texts in the Brown corpus from NLTK
corpus_english = brown.raw(categories="news")

# We eliminate all tags using regex and tokenize the text using NLTK
text_raw_english = re.sub(r"/[^\s]+", "", corpus_english)
tokens_english = word_tokenize(text_raw_english)

# We use the puntuation-symbols list from the string library
# We add two types of punctuation that the corpus also has
# We make all words lower case and check that they are not punctuation
# We make a list
punctuation = list(punctuation)
punctuation.append("``")
punctuation.append("''")
tokens_english = [w.lower() for w in tokens_english if w not in punctuation]


# This function will take the tokens of a corpus and the name (for the plot)
# It also has a parameter to exclude all words of length = 1
# It calculates the frequencies by word-length and fits a linear regression
# Makes a plot with that
# And returns several calculated values
def calculate_zipf(words, name, exclude_len1=False):

    # This is to exclude words of length=1 if the parameter is True
    if exclude_len1:
        words = [w for w in words if len(w) != 1]

    # We count the frequencies of each word using Counter
    word_frequencies = Counter(words)
    length_freq = {}

    # We calculate the total number of words
    total_words = len(words)

    # We aggregate frequencies by word length
    # By creating a dictionary with word-length as the key
    # And total number of occurences as the value
    for word, freq in word_frequencies.items():
        length = len(word)
        if length not in length_freq:
            length_freq[length] = 0
        length_freq[length] += freq

    # We convert both lengths and frequencyes to numpy arrays
    lengths = np.array(list(length_freq.keys()))
    frequencies = np.array(list(length_freq.values()))

    # We normalize the frequencies by total number of words
    normalized_frequencies = frequencies / total_words

    # We apply log10 to the normalized frequencies
    log_frequencies = np.log10(normalized_frequencies)

    # We fit a linear regression model using the lenghts and the lof of the freqs
    # And save the outputs of the model
    slope, intercept, r_value, p_value, std_err = linregress(lengths, log_frequencies)

    # We plot the data and the linear regression
    plt.figure(figsize=(12, 6))
    plt.scatter(lengths, log_frequencies, alpha=0.7, color="blue", label="Data")
    plt.plot(
        lengths,
        slope * lengths + intercept,
        color="green",
        linestyle="dashed",
        label="Linear Fit",
    )
    plt.title(f"Word Length vs Log(Frequency) with Linear Fit in {name}")
    plt.xlabel("Word Length")
    plt.ylabel("Log10(Frequency)")
    plt.legend()
    plt.grid(True)
    if save_plots:
        plt.savefig(f"{plot_save_adress}Length_Frequency_{name}.png")
    else:
        plt.show()
    plt.close()

    # We return a dictionary with different values that we calculated
    # To be able to use them later
    return {
        "lengths": lengths,
        "slope": slope,
        "r_squared": r_value**2,
        "p_value_regression": p_value,
        "frequencies": frequencies,
        "normalized_frequencies": normalized_frequencies,
        "total_words": total_words,
    }


# This function takes as parameters two dictionaries with corpus values
# Calculated using the calculate_zipf() function
# And also the names of the two corpora for  outputs
# It compares and outputs information about the behaviour of the corpora
def compare_corpora(corpus1_stats, corpus2_stats, name1="Corpus 1", name2="Corpus 2"):

    # This is just for aesthetics
    print(f"\nComparison of {name1} and {name2}:")
    print("-" * 40)

    # Compare the slopes of the linear regressions
    print(f"Slope (shorter words more frequent):")
    print(
        f"    {name1}: {corpus1_stats['slope']:.2f}, p-value: {corpus1_stats["p_value_regression"]:.4f}"
    )
    print(
        f"    {name2}: {corpus2_stats['slope']:.2f}, p-value: {corpus2_stats["p_value_regression"]:.4f}"
    )
    print("→ More negative slope means stronger Zipf's Law effect.\n")

    # Compare the R² values
    print(f"R² Value (Strength of Linear Fit):")
    print(f"    {name1}: {corpus1_stats['r_squared']:.2f}")
    print(f"    {name2}: {corpus2_stats['r_squared']:.2f}")
    print(
        "→ Higher R² means a stronger correlation between word length and frequency.\n"
    )

    # This two chunks unpack the two arrays we did and generate a list
    # That has as many instances of the word-length value as it was frequent
    # This is to be able to use it for the Kolmogorov-Smirnov test
    corpus1_samples = np.concatenate(
        [
            [length] * int(freq * corpus1_stats["total_words"])
            for length, freq in zip(
                corpus1_stats["lengths"], corpus1_stats["normalized_frequencies"]
            )
        ]
    )
    corpus2_samples = np.concatenate(
        [
            [length] * int(freq * corpus2_stats["total_words"])
            for length, freq in zip(
                corpus2_stats["lengths"], corpus2_stats["normalized_frequencies"]
            )
        ]
    )

    # Apply the Kolmogorov-Smirnov 2-sample test to check if the distributions are similar
    # And print the results
    ks2_stat, ks2_p = ks_2samp(corpus1_samples, corpus2_samples)
    print("Kolmogorov–Smirnov 2-Sample Test (Corpus vs Corpus):")
    print(f"    KS Statistic: {ks2_stat:.4f}, p-value: {ks2_p:.4f}")
    print("→ Higher KS stat means a bigger difference.\n")


# Obtain the plots and values for the both corpora
stats_english = calculate_zipf(tokens_english, "English")
stats_gaelic = calculate_zipf(tokens_gaelic, "Gaelic")

# Compare both corpora
compare_corpora(stats_english, stats_gaelic, name1="English", name2="Gaelic")

# Do the same but without taking into account words of length = 1
stats_english_no1 = calculate_zipf(tokens_english, "English without length-1", True)
stats_gaelic_no1 = calculate_zipf(tokens_gaelic, "Gaelic without length-1", True)
compare_corpora(
    stats_english_no1,
    stats_gaelic_no1,
    name1="English without length-1",
    name2="Gaelic without length-1",
)
