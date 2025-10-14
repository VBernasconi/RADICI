## Updated code to setup the database
Run the following commands
```
python redis_init.py
```
The above command will get the content of existing .csv file to setup a clean db on Redis
```
python get_new_json_entries.py
```
The above command will go through a jsonl file to import new entries into the Redis db
```
python redis_json_import.py
```
The above command will go through a geojson file with existing coordinates to update the Redis db
