# https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-ubuntu/
# https://www.mongodb.com/docs/manual/administration/production-notes/ LEER
sudo apt-get install gnupg curl

curl -fsSL https://pgp.mongodb.com/server-7.0.asc | \
   sudo gpg -o /usr/share/keyrings/mongodb-server-7.0.gpg \
   --dearmor

echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/7.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-7.0.list

sudo apt-get update

sudo apt-get install -y mongodb-org

# START MONGO
# sudo systemctl start mongod
# STOP MONGO
# sudo systemctl stop mongod
# REMOVE MONGO https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-ubuntu/