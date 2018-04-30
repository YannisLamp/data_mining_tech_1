from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from wordcloud import WordCloud
import sys
import pandas as pd
import matplotlib.pyplot as plot


def make_cloud(category_content_txt, categories, wordcloud):
    print category_content_txt
    for i,txt in enumerate(category_content_txt):
        print("wordcloud for:",categories[i])
        wordcloud.generate(txt)
        plot.imshow(wordcloud)
        plot.axis('off')
        filename = categories[i] + "Wordcloud"
        plot.savefig(filename)
        plot.show()


def main(argv):

    category_content_txt = []
    data = pd.read_csv('datasets/train_set.csv', sep="\t")
    all_categories = set(data['Category'])
    categories = ["Politics", "Film", "Football", "Business", "Technology"]

    stop_words = ENGLISH_STOP_WORDS
    font = '/usr/share/fonts/truetype/ubuntu-font-family/Ubuntu-B.ttf'

    for cat in categories:
        #cat_rows = data.lookup()
        cat_rows = data.loc[data['Category'] == cat]
        cat_content = cat_rows.loc[:, "Content"]
        cat_content_txt = cat_content.to_string()
        category_content_txt.append(cat_content_txt)

    wordcloud = WordCloud(
        max_words=50,
        stopwords=stop_words,
        background_color='white',
        width=1200,
        height=1000,
        font_path=font
    )

    make_cloud(category_content_txt, categories, wordcloud)


if __name__ == "__main__":
    main(sys.argv)
