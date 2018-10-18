# coding:utf-8
"""
爬这个货
http://www.tianqihoubao.com/lishi/chongqing/month/201806.html

参考这个链接
http://www.tianqihoubao.com/lishi/chongqing.html

下次再爬这个货
http://lishi.tianqi.com/chongqing/201101.html
"""
from urllib import request
from bs4 import BeautifulSoup
import re
import pandas as pd


def get_html(url_link):
    url_request = request.Request(str(url_link))
    html = request.urlopen(url_request).read()
    soup = BeautifulSoup(html, 'html.parser')
    ss = soup.find_all('tr')
    return ss

def get_data(soup_html):

    w1 = '<td>'
    w2 = '</td>'
    pat = re.compile(w1+'(.*?)'+w2, re.S)

    s1 = '>'
    s2 = '<'
    pat1 = re.compile(s1 + '(.*?)' + s2, re.S)

    final_list = []

    for i in range(1, len(soup_html)):
        result = pat.findall(str(soup_html[i]))
        for j in range(len(result)):
            result[j] = result[j].replace('\r\n', '')
            result[j] = result[j].replace(' ', '')

        sub_res_riqi = pat1.findall(result[0])[0]
        sub_res_tqzk_bai = result[1].split('/')[0]
        sub_res_tqzk_ye = result[1].split('/')[1]
        sub_res_qw_bai = result[2].split('/')[0]
        sub_res_qw_ye = result[2].split('/')[1]
        sub_res_flfx_bai = result[3].split('/')[0]
        sub_res_flfx_ye = result[3].split('/')[1]

        final_list.append([sub_res_riqi, sub_res_tqzk_bai, sub_res_tqzk_ye, sub_res_qw_bai, sub_res_qw_ye,
                           sub_res_flfx_bai, sub_res_flfx_ye])

    return final_list

column_list = ['日期', '天气状况(白)', '天气状况(夜)', '气温(白)', '气温(夜)',
               '风力方向(白)', '风力方向(夜)']
"""
'日期', '天气状况(白)','天气状况(夜)','气温(白)','气温(夜)',
'风力方向(白)', '风力方向(夜)'
"""

def get_dataframe(aarray_data, columns_names):
    df = pd.DataFrame(aarray_data, columns=columns_names)
    return df

def months_zhangxiao(columns=column_list):
    finals = []
    months_list = [201101, 201102, 201103, 201104, 201105, 201106, 201107, 201108, 201109, 201110, 201111, 201112,
                   201201, 201202, 201203, 201204, 201205, 201206, 201207, 201208, 201209, 201210, 201211, 201212,
                   201301, 201302, 201303, 201304, 201305, 201306, 201307, 201308, 201309, 201310, 201311, 201312,
                   201401, 201402, 201403, 201404, 201405, 201406, 201407, 201408, 201409, 201410, 201411, 201412]

    for month in months_list:
        month_url = 'http://www.tianqihoubao.com/lishi/chongqing/month/' + str(month) + '.html'
        tmp_html = get_html(month_url)
        month_result = get_data(tmp_html)
        print(month_result)
        finals += month_result
    final_df = get_dataframe(finals, columns)
    final_df.to_csv('tianqi_data_zhangxiao.csv', index=False)


def months_meijian(columns=column_list):
    finals = []
    months_list = [201501, 201502, 201503, 201504, 201505, 201506, 201507, 201508, 201509, 201510, 201511, 201512,
                   201601, 201602, 201603, 201604, 201605, 201606, 201607, 201608, 201609, 201610, 201611, 201612,
                   201701, 201702, 201703, 201704, 201705, 201706, 201707, 201708, 201709, 201710, 201711, 201712,
                   201801, 201802, 201803, 201804, 201805, 201806, 201807, 201808, 201809]
    for month in months_list:
        month_url = 'http://www.tianqihoubao.com/lishi/chongqing/month/' + str(month) + '.html'
        tmp_html = get_html(month_url)
        month_result = get_data(tmp_html)
        finals += month_result
    final_df = get_dataframe(finals, columns)
    final_df.to_csv('tianqi_data_meijian.csv', index=False)

if __name__ == '__main__':
    # s_html = get_html('http://www.tianqihoubao.com/lishi/chongqing/month/201806.html')
    # finals = get_data(s_html)
    # df1 = get_dataframe(finals, column_list)
    #
    # ss2 = 'http://www.tianqihoubao.com/lishi/chongqing/month/' + str(201805) + '.html'
    # ss2_htmal = get_html(ss2)
    # finals2 = get_data(ss2_htmal)
    print('a')
    months_zhangxiao()