from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from wordcloud import WordCloud
import sys
import pandas as pd
import matplotlib.pyplot as plt


def get_font():
    font = '/usr/share/fonts/truetype/ubuntu-font-family/Ubuntu-B.ttf'
    return font

def render_wordcloud(category_content_txt, categories, wordcloud):
    for i,txt in enumerate(category_content_txt):
        print("NOW PRINTING CLOUDWORD FOR CATEGORY:",categories[i])
        wordcloud.generate(txt)
        plt.imshow(wordcloud)
        plt.axis('off')
        filename = "WordCloud-" + categories[i]
        plt.savefig(filename)
        plt.show()

def main(argv):

    category_content_txt = []
    data = pd.read_csv('datasets/train_set.csv', sep="\t")
    all_categories = set(data['Category'])

    #stop_words_temp = ['say', 'says', 'said','day', 'year', 'It', 'like', 'set','it', 'th']
    stop_words_temp = []
    stop_words = ENGLISH_STOP_WORDS.union(stop_words_temp)
    font = get_font()

    for cat in categories:
        cat_rows = df.lookup()
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

    render_wordcloud(category_content_txt, categories, wordcloud)


if __name__ == "__main__":
    main(sys.argv)
