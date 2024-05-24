FROM node:10

WORKDIR /main

COPY package*.json ./

RUN pip install requirements.txt

RUN npm install

COPY . .

EXPOSE 8080

CMD ["node","server.js"]
