from __future__ import annotations

from pydantic import BaseModel, Field


class ChatCompletionRequest(BaseModel):
    username: str = Field(..., examples=['andrea'], description="The username of the user making the request.")
    chat_id: str = Field(..., examples=['chat1'], description="The chat id of the messages")

    root_model: str = Field(..., examples=['openrouter/free'], description="The root model: the planner, the one that does the heavy lifting")
    sub_model: str = Field(..., examples=['openrouter/free'], description="The sub model: the one that is used by the mcp tool for text summarization or other text related stuff")
    openrouter_api_key: str = Field(..., examples=['sk-or-v1-xxx'], description="The OpenRouter API key to use for the models")

    messages: str | list[dict] = Field(..., examples=[[{'role':'user', 'content':'Summarize the content of the document'}]], description="The messages sent by the user in the conversation.")
    # In the future this could become a list of documents
    document: str = Field(..., examples=["""
Sempre caro mi fu quest’ermo colle,
e questa siepe, che da tanta parte
dell’ultimo orizzonte il guardo esclude.
Ma sedendo e mirando, interminati
spazi di là da quella, e sovrumani
silenzi, e profondissima quïete
io nel pensier mi fingo, ove per poco
il cor non si spaura. E come il vento
odo stormir tra queste piante, io quello
infinito silenzio a questa voce
vo comparando: e mi sovvien l’eterno,
e le morte stagioni, e la presente
e viva, e il suon di lei. Così tra questa
immensità s’annega il pensier mio:
e il naufragar m’è dolce in questo mare.
    """.strip()], description="A long document to be analyzed.")