import clickhouse_connect

if __name__ == '__main__':
    client = clickhouse_connect.get_client(
        host='cvzq3t560s.ap-southeast-1.aws.clickhouse.cloud',
        user='default',
        password='8~0lxNgJPB65E',
        secure=True
    )
    print("Result:", client.query("SELECT 1").result_set[0][0])
