__authors__ = "Nicholas Leonard"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Nicholas Leonard"]
__license__ = "3-clause BSD"
__maintainer__ = "Nicholas Leonard"
__email__ = "leonardn@iro"

import psycopg2, time, getpass

#For backwards compatability:
EXECUTE = 0
FETCH_ONE = 1
FETCH_ALL = 2
COMMIT = 3
COMMIT_AND_FETCH_ONE = 4
#End Deprecated

'''TO DO
- dynamic SQL Function file execution with changes
'''

"""DATABASE = raw_input("Enter name of Database (udem): ")
USER = raw_input("Enter name of database User (nicholas): ")
HOST = raw_input("Enter name of Host (nps): ")
PASSWORD = getpass.getpass("PostgreSQL password: ")
if USER is None or USER == '':
	USER = 'nicholas'
if HOST is None or HOST == '':
	HOST = 'nps'
if DATABASE is None or DATABASE == '':
	DATABASE = 'udem' """    
USER = 'nicholas'
HOST = 'nikopia.net'
DATABASE = 'udem'
PASSWORD = ''
class DatabaseHandler:
	DatabaseError = psycopg2.DatabaseError
	EXECUTE = 0
	FETCH_ONE = 1
	FETCH_ALL = 2
	COMMIT = 3
     
	def __init__(self, database=DATABASE, user=USER, host=HOST, password=PASSWORD):
		self.conn = psycopg2.connect(database=database, user=user, host=host, password=password)
		c = self.conn.cursor()
		c.execute("SET CLIENT_ENCODING to 'UNICODE';")
		#c.execute("SET AUTOCOMMIT TO ON;")
		self.conn.commit()
		c.close()
	def executeSQL(self, command, param=False, action=COMMIT):
		result = None
		c = self.conn.cursor()
		start = time.time()
		try:
			if not param:
				c.execute(command)
			else:
				c.execute(command, param)
			if action == EXECUTE:
				c.close()
				result = True
			elif action == self.FETCH_ONE:
				row = c.fetchone()
				self.conn.commit()
				c.close()
				self.check(command, start)
				result = row
			elif action == self.FETCH_ALL:
				rows = []
				row = c.fetchone()
				while row:
				    rows.append(row)
				    row = c.fetchone()
				self.conn.commit()
				c.close()
				self.check(command, start)
				result = rows
			elif action == self.COMMIT:
				self.conn.commit()
				c.close()
				result = True
		except DatabaseHandler.DatabaseError, e:
			print "DatabaseHandler.executeSQL() failed for command:", command, param, action, e
			self.conn.rollback()
			c.close()
			result = False
		#self.dbLock.release()
		return result
	def check(self, command, start):
		if time.time()-start > 60:
			print command, time.time()-start
			self.conn.close()
			DatabaseHandler.__init__(self)


