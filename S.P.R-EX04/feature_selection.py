import utilities as utl


if __name__ == '__main__':
    # anneal.txt
    anneal_x, anneal_y = utl.load_data('./datasets/anneal.txt')
    anneal_tesult = utl.result(anneal_x, anneal_y, 'anneal.txt')
    print(anneal_tesult)

    print()

    # diabetes.txt
    diabetes_x, diabetes_y = utl.load_data('./datasets/anneal.txt')
    diabetes_result = utl.result(diabetes_x, diabetes_y, 'diabetes.txt')
    print(diabetes_result)

    print()

    # hepatitis.txt
    hepatitis_x, hepatitis_y = utl.load_data('./datasets/hepatitis.txt')
    hepatitis_result = utl.result(hepatitis_x, hepatitis_y, 'hepatitis.txt')
    print(hepatitis_result)

    print()

    # kr-vs-kp.txt
    krkp_x, krkp_y = utl.load_data('./datasets/kr-vs-kp.txt')
    krkp_result = utl.result(krkp_x, krkp_y, 'kr-vs-kp.txt')
    print(krkp_result)

    print()

    # vote.txt
    vote_x, vote_y = utl.load_data('./datasets/vote.txt')
    vote_result = utl.result(vote_x, vote_y, 'vote.txt')
    print(vote_result)
