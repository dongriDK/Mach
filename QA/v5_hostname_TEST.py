import v5_TEST

def hostname_test(api_key):
    v5_hostname_filepath = "./test_list/hostname_list.txt"
    hostname_T = v5_TEST.v5_TEST(api_type=2,path=v5_hostname_filepath, api_key=api_key,api_url='')
    hostname_T.v5_request()
