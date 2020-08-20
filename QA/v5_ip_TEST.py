import v5_TEST

def ip_test(api_key):
    v5_ip_filepath = "./test_list/ip_list.txt"
    ip_T = v5_TEST.v5_TEST(api_type=1,path=v5_ip_filepath, api_key=api_key,api_url='')
    ip_T.v5_request()
