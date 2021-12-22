import pymysql

# connection & cursor
conn = pymysql.connect(host="localhost",
                       user="root",
                       password='Dhqxlvmfkdla11@',
                       charset='utf8'
                       )
cur = conn.cursor()

# create a db and a table
cur.execute("CREATE DATABASE rc_googlemovies")
cur.execute("USE rc_googlemovies")
cur.execute("""CREATE TABLE contentsbased (
                    movie_id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
                    title VARCHAR(100) NOT NULL,
                    genre VARCHAR(50) NOT NULL,
                    overview LONGTEXT,
                    rate VARCHAR(10) NOT NULL
                    )""")

conn.commit()
conn.close()
