FROM node:14.15.0-alpine

# EXPOSE 4200

WORKDIR /usr/src/frontend

COPY ./package*.json ./
RUN npm install -g @angular/cli
RUN npm install

COPY . ./

CMD ["ng", "serve", "--host", "0.0.0.0"]

# CMD ["npm", "start"]