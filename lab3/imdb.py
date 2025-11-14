"""
install dependencies:
pip install -r requirements.txt

run gui.py to start the application:
python gui.py

AUTHORS: Kamil Suchomski and Kamil Koniak

PROBLEM: The task was to create a recommendation engine for movies and series based on the data collected from survey.

SOLUTION: We used Item based Collaborative Filtering with Cosine Similarity algorithm to create the recommendation engine.
We also used IMDb API to get the data about the movies and series, to display the posters, ratings, genres, duration, and descriptions.


This module provides a graphical user interface for the recommendation system,
allowing users to:
- Select a user from a dropdown menu
- View personalized movie recommendations (likely to enjoy) and anti-recommendations (likely to skip)
- Browse detailed IMDb information for recommended movies
- See movie posters, ratings, genres, duration, and descriptions

The GUI features a modern dark red theme with beautiful typography and responsive
design. IMDb data is fetched asynchronously to keep the UI responsive.

Fetch and display IMDb data for a given title id.
"""

from __future__ import annotations

import io

import requests
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

API_URL = "https://imdb.iamidiotareyoutoo.com/search"


def fetch_imdb_data(imdb_id: str) -> dict:
    """Call the IMDb proxy API and return the JSON payload."""
    if not imdb_id:
        raise ValueError("IMDb id is required.")

    response = requests.get(API_URL, params={"tt": imdb_id}, timeout=15)
    response.raise_for_status()
    payload = response.json()
    if not payload.get("ok"):
        raise RuntimeError(f"IMDb API error: {payload.get('description')}")
    return payload


def load_image_from_url(url: str, max_width: int = 400) -> ImageTk.PhotoImage | None:
    if not url:
        return None
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))
        if image.width > max_width:
            ratio = max_width / float(image.width)
            new_size = (max_width, int(image.height * ratio))
            image = image.resize(new_size, Image.LANCZOS)
        return ImageTk.PhotoImage(image)
    except Exception:  # noqa: BLE001
        return None


def show_imdb_window(imdb_id: str) -> None:
    """Open a Tkinter window with IMDb data for the given id."""
    try:
        payload = fetch_imdb_data(imdb_id)
    except Exception as exc:  # noqa: BLE001
        messagebox.showerror("IMDb Error", f"Could not load IMDb data:\n{exc}")
        return

    short = payload.get("short", {})
    name = short.get("name", "Unknown title")
    alternate_name = short.get("alternateName", "")
    description = short.get("description", "No description available.")
    genres = short.get("genre", [])
    duration = short.get("duration", "")
    aggregate = short.get("aggregateRating", {})

    window = tk.Toplevel()
    window.title(f"IMDb – {name}")
    window.geometry("600x500")

    container = ttk.Frame(window, padding=10)
    container.pack(fill="both", expand=True)

    info_frame = ttk.Frame(container)
    info_frame.pack(fill="both", expand=True)

    image_url = short.get("image")
    poster = load_image_from_url(image_url)
    if poster:
        label = ttk.Label(info_frame, image=poster)
        label.image = poster
        label.pack(side="left", padx=10)

    text_frame = ttk.Frame(info_frame)
    text_frame.pack(side="left", fill="both", expand=True, padx=10)

    ttk.Label(text_frame, text=name, font=("Segoe UI", 14, "bold")).pack(anchor="w")
    if alternate_name and alternate_name != name:
        ttk.Label(text_frame, text=f"Alternate: {alternate_name}", font=("Segoe UI", 10, "italic")).pack(anchor="w")

    if genres:
        ttk.Label(text_frame, text=f"Genres: {', '.join(genres)}").pack(anchor="w", pady=(4, 0))
    if duration:
        ttk.Label(text_frame, text=f"Duration: {duration}").pack(anchor="w")

    agg_text = ""
    if aggregate:
        rating = aggregate.get("ratingValue")
        votes = aggregate.get("ratingCount")
        agg_text = f"Rating: {rating} / {aggregate.get('bestRating', 10)} ({votes} votes)"
    ttk.Label(text_frame, text=agg_text or "Rating information not available").pack(anchor="w", pady=(4, 0))

    desc_box = tk.Text(container, wrap="word", height=8)
    desc_box.pack(fill="both", expand=True, pady=10)
    desc_box.insert("1.0", description)
    desc_box.configure(state="disabled")


def main() -> None:
    root = tk.Tk()
    root.withdraw()
    demo_id = "tt0314134"
    show_imdb_window(demo_id)
    root.mainloop()


if __name__ == "__main__":
    main()

