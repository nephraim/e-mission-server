# Standard imports
from datetime import datetime, timedelta
import json

# Our imports
from emission.core.get_database import get_client_db, get_db, get_section_db

def makeValid(client):
  client.clientJSON['start_date'] = str(datetime.now() + timedelta(days=-2))
  client.clientJSON['end_date'] = str(datetime.now() + timedelta(days=+2))
  # print (client.clientJSON)
  client._Client__update(client.clientJSON)

def makeExpired(client):
  client.clientJSON['start_date'] = str(datetime.now() + timedelta(days=-4))
  client.clientJSON['end_date'] = str(datetime.now() + timedelta(days=-2))
  # print (client.clientJSON)
  client._Client__update(client.clientJSON)

def updateUserCreateTime(uuid):
  from emission.core.wrapper.user import User
  
  user = User.fromUUID(uuid)
  user.changeUpdateTs(timedelta(days = -20))

def dropAllCollections(db):
  collections = db.collection_names()
  print "collections = %s" % collections
  for coll in collections:
    if coll.startswith('system'):
      print "Skipping system collection %s" % coll
    else: 
      print "Dropping collection %s" % coll
      db.drop_collection(coll)

def purgeSectionData(Sections, userName):
    """
    Deletes all sections for this user.
    TODO: Need to extend it to delete entries across all collections
    """
    Sections.remove({'user_id' : userName})

def loadTable(serverName, tableName, fileName):
  tableColl = get_db()[tableName]
  dataJSON = json.load(open(fileName))
  for row in dataJSON:
    tableColl.insert(row)

# Create a dummy section with the main stuff that we use in our code
def createDummySection(startTime, endTime, startLoc, endLoc, predictedMode = None, confirmedMode = None):
  section = {
             'source': 'Shankari',
             'section_start_datetime': startTime,
             'section_end_datetime': endTime,
             'section_start_time': startTime.isoformat(),
             'section_end_time': endTime.isoformat(),
             'section_start_point': {'type': 'Point', 'coordinates': startLoc},
             'section_end_point': {'type': 'Point', 'coordinates': endLoc},
            }
  if predictedMode != None:
    section['predicted_mode'] = predictedMode
  if confirmedMode != None:
    section['confirmed_mode'] = confirmedMode

  get_section_db().insert(section)
  return section

def updateSections(testCase):
    from emission.core.wrapper.user import User
    """
    Updates sections with appropriate test data
    Should be called anytime new data is loaded into the
    'Stage_Sections' table
    """
    testCase.uuid_list = []
    for section in testCase.SectionsColl.find():
      section['section_start_datetime'] = testCase.dayago
      section['section_end_datetime'] = testCase.dayago + timedelta(hours = 1)
      section['predicted_mode'] = [0, 0.4, 0.6, 0]
      section['confirmed_mode'] = ''
      # Replace the user email with the UUID
      curr_uuid = User.fromEmail(section['user_id']).uuid
      section['user_id'] = curr_uuid
      testCase.uuid_list.append(curr_uuid)
      testCase.SectionsColl.save(section)
