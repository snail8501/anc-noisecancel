package main

import (
	"fmt"
	"log"
	"net/http"

	"github.com/gordonklaus/portaudio"
	"github.com/gorilla/websocket"
)

var upgrader = websocket.Upgrader{}
var clients = make(map[*websocket.Conn]bool)

func main() {
	portaudio.Initialize()
	defer portaudio.Terminate()

	buffer := make([]int16, 512)
	stream, err := portaudio.OpenDefaultStream(1, 0, 44100, len(buffer), buffer)
	if err != nil {
		log.Fatal(err)
	}
	defer stream.Close()

	// Serve HTML page at http://localhost:8080/
	fs := http.FileServer(http.Dir("./public"))
	http.Handle("/", fs)

	// WebSocket endpoint
	http.HandleFunc("/ws", handleWS)

	go http.ListenAndServe(":8080", nil)
	fmt.Println("Server started at http://localhost:8080/")

	stream.Start()
	defer stream.Stop()

	for {
		if err := stream.Read(); err != nil {
			log.Println("Audio read error:", err)
			continue
		}

		for c := range clients {
			err := c.WriteMessage(websocket.BinaryMessage, int16SliceToBytes(buffer))
			if err != nil {
				log.Println("WebSocket write error:", err)
				c.Close()
				delete(clients, c)
			}
		}
	}
}

func handleWS(w http.ResponseWriter, r *http.Request) {
	upgrader.CheckOrigin = func(r *http.Request) bool { return true }
	ws, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Println("Upgrade error:", err)
		return
	}
	clients[ws] = true
}

func int16SliceToBytes(data []int16) []byte {
	b := make([]byte, len(data)*2)
	for i, v := range data {
		b[2*i] = byte(v)
		b[2*i+1] = byte(v >> 8)
	}
	return b
}
