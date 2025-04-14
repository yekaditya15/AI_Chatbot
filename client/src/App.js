import React, { useState, useRef, useEffect, useCallback } from "react";
import {
  AppBar,
  Box,
  Container,
  Typography,
  Paper,
  TextField,
  Button,
  IconButton,
  List,
  ListItem,
  ListItemText,
  Toolbar,
  Card,
  CardContent,
  Chip,
} from "@mui/material";
import {
  Send as SendIcon,
  CloudUpload as CloudUploadIcon,
  SmartToy as SmartToyIcon,
  Delete as DeleteIcon,
} from "@mui/icons-material";

function App() {
  const [file, setFile] = useState(null);
  const [fileName, setFileName] = useState("No file chosen");
  const [uploadStatus, setUploadStatus] = useState("");
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const fileInputRef = useRef(null);
  const messagesEndRef = useRef(null);
  const [isDocumentMode, setIsDocumentMode] = useState(false);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  const handleFileChange = (e) => {
    if (e.target.files[0]) {
      setFile(e.target.files[0]);
      setFileName(e.target.files[0].name);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    const formData = new FormData();
    formData.append("file", file);

    try {
      setUploadStatus("Uploading...");
      const response = await fetch(
        "https://ai-chatbot-g10o.onrender.com/upload",
        {
          method: "POST",
          body: formData,
        }
      );
      const data = await response.json();
      if (response.ok) {
        setUploadStatus(data.message);
        setMessages([]); // Clear chat history when new document is uploaded
      } else {
        setUploadStatus("Error: " + data.detail);
      }
    } catch (error) {
      setUploadStatus("Upload failed: " + error.message);
    }
  };

  const clearDocument = async () => {
    try {
      await fetch("https://ai-chatbot-g10o.onrender.com/clear", {
        method: "POST",
      });
      setFile(null);
      setFileName("No file chosen");
      setUploadStatus("");
      setMessages([]); // Clear chat history when document is cleared
    } catch (error) {
      console.error("Error clearing document mode:", error);
    }
  };

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMessage = input;
    setInput("");
    setMessages((prev) => [...prev, { sender: "user", text: userMessage }]);
    setLoading(true);

    try {
      const response = await fetch(
        "https://ai-chatbot-g10o.onrender.com/chat",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question: userMessage }),
        }
      );
      const data = await response.json();
      setMessages((prev) => [
        ...prev,
        {
          sender: "bot",
          text: data.answer,
          mode: data.mode, // 'document' or 'chat'
        },
      ]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        { sender: "bot", text: "Error: " + error.message },
      ]);
    }
    setLoading(false);
  };

  const checkMode = async () => {
    try {
      const response = await fetch("https://ai-chatbot-g10o.onrender.com/mode");
      const data = await response.json();
      setIsDocumentMode(data.mode === "document");
    } catch (error) {
      console.error("Error checking mode:", error);
    }
  };

  useEffect(() => {
    checkMode();
  }, []);

  return (
    <Box sx={{ flexGrow: 1 }}>
      <AppBar position="static">
        <Toolbar>
          <SmartToyIcon sx={{ mr: 2 }} />
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            AI Assistant
          </Typography>
        </Toolbar>
      </AppBar>

      <Container maxWidth="md" sx={{ mt: 4 }}>
        <Card sx={{ mb: 4 }}>
          <CardContent>
            <Typography variant="h5" gutterBottom>
              Upload Document (Optional)
            </Typography>
            <Box sx={{ display: "flex", alignItems: "center", gap: 2, mb: 2 }}>
              <input
                type="file"
                ref={fileInputRef}
                style={{ display: "none" }}
                onChange={handleFileChange}
              />
              <Button
                variant="contained"
                startIcon={<CloudUploadIcon />}
                onClick={() => fileInputRef.current.click()}
              >
                Choose File
              </Button>
              <Typography>{fileName}</Typography>
              {file && (
                <IconButton color="error" onClick={clearDocument}>
                  <DeleteIcon />
                </IconButton>
              )}
            </Box>
            {file && (
              <Button
                variant="contained"
                color="primary"
                onClick={handleUpload}
                fullWidth
              >
                Upload Document
              </Button>
            )}
            {uploadStatus && (
              <Typography sx={{ mt: 2, color: "text.secondary" }}>
                {uploadStatus}
              </Typography>
            )}
          </CardContent>
        </Card>

        <Paper
          elevation={3}
          sx={{
            height: "60vh",
            display: "flex",
            flexDirection: "column",
          }}
        >
          <Box
            sx={{
              flex: 1,
              overflowY: "auto",
              p: 2,
              backgroundColor: "#f5f5f5",
            }}
          >
            <List>
              {messages.map((msg, index) => (
                <ListItem
                  key={index}
                  sx={{
                    justifyContent:
                      msg.sender === "user" ? "flex-end" : "flex-start",
                    mb: 1,
                  }}
                >
                  <Paper
                    sx={{
                      maxWidth: "70%",
                      p: 2,
                      backgroundColor:
                        msg.sender === "user" ? "#e3f2fd" : "#fff",
                    }}
                  >
                    <ListItemText
                      primary={
                        <Typography variant="body1">{msg.text}</Typography>
                      }
                      secondary={
                        <Box sx={{ mt: 1 }}>
                          <Chip
                            label={msg.sender === "user" ? "You" : "AI"}
                            size="small"
                            color={
                              msg.sender === "user" ? "primary" : "secondary"
                            }
                            sx={{ mr: 1 }}
                          />
                          {msg.sender === "bot" && msg.mode && (
                            <Chip
                              label={
                                msg.mode === "document" ? "Document" : "Chat"
                              }
                              size="small"
                              color="info"
                            />
                          )}
                        </Box>
                      }
                    />
                  </Paper>
                </ListItem>
              ))}
              <div ref={messagesEndRef} />
            </List>
          </Box>

          <Box sx={{ p: 2, backgroundColor: "#fff" }}>
            <Box sx={{ display: "flex", gap: 1 }}>
              <TextField
                fullWidth
                variant="outlined"
                placeholder="Type your message..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={(e) => {
                  if (e.key === "Enter") sendMessage();
                }}
              />
              <IconButton
                color="primary"
                onClick={sendMessage}
                disabled={!input.trim() || loading}
              >
                <SendIcon />
              </IconButton>
            </Box>
          </Box>
        </Paper>
      </Container>
    </Box>
  );
}

export default App;
