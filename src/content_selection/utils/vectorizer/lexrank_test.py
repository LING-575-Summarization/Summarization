from content_selection import LexRankFactory

def main():
    LexRank = LexRankFactory('bert')
    x = LexRank.from_data('D1001A-A', 'data/devtest.json', sentences_are_documents=True,
                          min_length=5, min_jaccard_dist=0.5)
    result = x.solve_lexrank()
    print(result)
    print(x.obtain_summary())

if __name__ == '__main__':
    main()