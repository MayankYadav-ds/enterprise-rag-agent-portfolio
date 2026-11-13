FROM node:22.22.2-alpine3.23 AS build

WORKDIR /frontend
COPY package.json package-lock.json ./
RUN npm ci
COPY . ./
RUN npm run build

FROM caddy:2.11.8-alpine
COPY Caddyfile /etc/caddy/Caddyfile
COPY --from=build /frontend/dist /srv
EXPOSE 80 443 443/udp
