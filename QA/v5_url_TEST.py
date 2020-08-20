import v5_TEST

def url_test(api_key):
    v5_url_filepath = "./test_list/url_list.txt"
    url_T = v5_TEST.v5_TEST(api_type=3,path=v5_url_filepath, api_key=api_key,api_url='')
    url_T.v5_request()
