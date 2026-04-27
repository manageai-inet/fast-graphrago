FROM golang:1.25-alpine AS builder

WORKDIR /src

COPY go.mod go.sum ./
COPY example/go.mod example/go.sum ./example/

WORKDIR /src/example
RUN go mod download

WORKDIR /src
COPY . ./

WORKDIR /src/example
RUN CGO_ENABLED=0 GOOS=linux go build -o /out/fastgraphrago-example .

FROM alpine:3.22

WORKDIR /app

RUN apk add --no-cache ca-certificates

COPY --from=builder /out/fastgraphrago-example /app/fastgraphrago-example
COPY example/example.env /app/.env

EXPOSE 3000

CMD ["/app/fastgraphrago-example"]
