import gradio as gr
from moviebot import MovieBot

bot = MovieBot()

def respond(message, history):
    reply = bot.chat(message)
    return [
        {"role": "assistant", "content": reply}
    ]

chat_interface = gr.ChatInterface(
    fn= respond,
    title= "🎬 MovieBot Assistant",
    description= "Ask about movie plots, characters, genre, etc!",
    #theme= "soft",
    type= "messages",
    css="""
    .gradio-container {
        background-color: #878787 !important;  /* Black background */
        color: white !important;               /* White text generally */
    }
    """
)

if __name__ == "__main__":
    chat_interface.launch(share = True, inbrowser= True)
