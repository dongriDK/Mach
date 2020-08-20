import v5_TEST

def hash_test(api_key):
    v5_hash_filepath = "./test_list/hash_list.txt"
    hash_T = v5_TEST.v5_TEST(api_type=0,path=v5_hash_filepath, api_key=api_key,api_url='')
    hash_T.v5_request()
