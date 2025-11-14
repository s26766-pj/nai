"""
install dependencies:
pip install -r requirements.txt

run gui.py to start the application:
python gui.py

AUTHORS: Kamil Suchomski and Kamil Koniak

PROBLEM: The task was to create a recommendation engine for movies and series based on the data collected from survey.

SOLUTION: We used Item based Collaborative Filtering with Cosine Similarity algorithm to create the recommendation engine.
We also used IMDb API to get the data about the movies and series, to display the posters, ratings, genres, duration, and descriptions.
"""

from pathlib import Path
import html
import re
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Dict, List, Tuple

import pandas as pd

from imdb import fetch_imdb_data, load_image_from_url
from helper import (
    format_recommendations_with_metadata,
    load_datasets,
    plot_input_data,
    plot_output_data,
    stratified_user_split,
    evaluate_model,
)
from recommender import ItemBasedCollaborativeFilteringWithCosineSimilarity


class RecommendationGUI:
    """
    Main GUI class for the movie recommendation system.
    
    This class manages the entire user interface, including:
    - Loading and managing data (movies, ratings, users, IMDb links)
    - Training and using the recommendation model
    - Displaying recommendations and IMDb details
    - Handling user interactions
    """

    def __init__(self, root: tk.Tk) -> None:
        """
        Initialize the recommendation GUI application.
        
        Sets up the main window, loads all necessary data, trains the recommendation
        model, configures the theme, and builds the user interface widgets.
        
        Args:
            root: The root Tkinter window instance
        """
        # Configure main window properties
        self.root = root
        self.root.title("Recommendations by Kamil Suchomski and Kamil Koniak - Item based Collaborative Filtering with Cosine Similarity")
        self.root.geometry("1100x700")  # Initial window size
        self.root.minsize(900, 600)  # Minimum window size
        self._center_window()  # Center window on screen

        # Get the base directory for data files
        self.base_dir = Path(__file__).resolve().parent
        
        # Load datasets: movies metadata and user ratings
        self.movies, self.ratings = load_datasets(self.base_dir)
        
        # Load additional data files
        self.users_df = pd.read_csv(self.base_dir / "data" / "users.csv")
        self.links_df = pd.read_csv(self.base_dir / "data" / "links.csv")

        # Create lookup dictionaries for fast access
        # Map movieId -> title for quick title lookups
        self.movie_lookup = self.movies.set_index("movieId")["title"].to_dict()
        # Map movieId -> imdbId for IMDb API calls
        self.imdb_lookup = self.links_df.set_index("movieId")["imdbId"].to_dict()
        # Map userId -> userName for displaying user names
        self.user_lookup = self.users_df.set_index("userId")["userName"].to_dict()
        
        # Cache for IMDb data to avoid redundant API calls
        self.imdb_cache: Dict[str, dict] = {}

        # Initialize and train the recommendation model
        # Using hybrid collaborative filtering with content-based features
        self.model = ItemBasedCollaborativeFilteringWithCosineSimilarity(
            top_k_similar=25,      # Number of similar items to consider
            min_similarity=0.05,   # Minimum similarity threshold
            content_weight=0.3     # 30% content features, 70% collaborative filtering
        )
        # Train on the full dataset for best recommendations
        self.model.fit(self.ratings, movies=self.movies)

        # Evaluate model performance on a test split (for status bar display)
        split = stratified_user_split(self.ratings, test_ratio=0.2, random_state=42)
        self.metrics = evaluate_model(self.model, split.test)

        # Configure the visual theme (dark red color scheme)
        self._configure_theme()
        # Build all UI widgets
        self._build_widgets()

    def _center_window(self) -> None:
        """
        Center the main window on the screen.
        
        Calculates the screen center position and updates the window geometry
        to place it at the center of the display.
        """
        # Window dimensions (matching the geometry set in __init__)
        width, height = 1100, 700
        
        # Get screen dimensions using Tkinter's screen info methods
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Calculate center position: screen center minus half window size
        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)
        
        # Update window geometry: format is "widthxheight+x+y"
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def _configure_theme(self) -> None:
        """
        Configure the visual theme with a modern dark red color scheme.
        
        Sets up all widget styles including colors, fonts, and interactive states.
        Uses the 'clam' theme as a base for better customization control.
        Stores theme colors and fonts as instance variables for reuse.
        """
        # Dark red color scheme - carefully chosen for readability and aesthetics
        bg_dark_red = "#2d0a0a"  # Very dark red background (main window)
        bg_medium_red = "#4a1414"  # Medium dark red for frames and panels
        fg_light = "#f5f5f5"  # Light text color for contrast
        accent_red = "#c41e3a"  # Bright red for selections and highlights
        hover_red = "#8b1538"  # Darker red for hover states
        
        # Configure root window background color
        self.root.configure(bg=bg_dark_red)
        
        # Initialize ttk Style object for theme customization
        style = ttk.Style()
        # Use 'clam' theme as base - it's more customizable than default themes
        style.theme_use("clam")
        
        # Modern fonts
        font_heading = ("Segoe UI", 12, "bold")
        font_body = ("Segoe UI", 10)
        font_small = ("Segoe UI", 9)
        
        # Configure Frame styles
        style.configure("TFrame", background=bg_dark_red)
        style.configure("TLabelframe", background=bg_medium_red, foreground=fg_light, 
                       borderwidth=2, relief="flat")
        style.configure("TLabelframe.Label", background=bg_medium_red, foreground=fg_light,
                       font=font_heading)
        
        # Configure Label styles
        style.configure("TLabel", background=bg_dark_red, foreground=fg_light, font=font_body)
        style.map("TLabel", background=[("active", bg_dark_red)])
        
        # Configure Combobox style
        style.configure("TCombobox", fieldbackground=bg_medium_red, background=bg_medium_red,
                       foreground=fg_light, borderwidth=1, font=font_body)
        style.map("TCombobox",
                 fieldbackground=[("readonly", bg_medium_red)],
                 selectbackground=[("readonly", accent_red)],
                 selectforeground=[("readonly", fg_light)])
        
        # Configure Button style (if needed later)
        style.configure("TButton", background=bg_medium_red, foreground=fg_light,
                       font=font_body, borderwidth=1, relief="flat")
        style.map("TButton",
                 background=[("active", hover_red), ("pressed", accent_red)],
                 foreground=[("active", fg_light)])
        
        # Configure Treeview (for the movie lists)
        style.configure("Treeview", background=bg_medium_red, foreground=fg_light,
                       fieldbackground=bg_medium_red, font=font_body, borderwidth=1)
        style.configure("Treeview.Heading", background=bg_dark_red, foreground=fg_light,
                       font=font_heading, relief="flat")
        style.map("Treeview",
                 background=[("selected", accent_red)],
                 foreground=[("selected", fg_light)])
        
        # Configure Scrollbar
        style.configure("TScrollbar", background=bg_medium_red, troughcolor=bg_dark_red,
                       borderwidth=0, arrowcolor=fg_light, darkcolor=bg_medium_red,
                       lightcolor=bg_medium_red)
        style.map("TScrollbar",
                 background=[("active", hover_red)],
                 arrowcolor=[("active", accent_red)])
        
        # Store colors and fonts for later use
        self.bg_dark_red = bg_dark_red
        self.bg_medium_red = bg_medium_red
        self.fg_light = fg_light
        self.accent_red = accent_red
        self.font_heading = font_heading
        self.font_body = font_body
        self.font_small = font_small

    def _build_widgets(self) -> None:
        """
        Build and arrange all GUI widgets.
        
        Creates the complete user interface including:
        - Top frame with user selection dropdown
        - User info display label
        - Left panel with recommendation tables (Likely to Enjoy / Likely to Skip)
        - Right panel with IMDb details (poster, title, metadata, description)
        - Bottom status bar with model metrics
        
        Automatically displays data for the first user if available.
        """
        # Top frame: contains user selection controls
        top_frame = ttk.Frame(self.root, padding=10)
        top_frame.pack(fill="x")  # Fill horizontally, don't expand vertically

        # Label for user selection
        ttk.Label(
            top_frame,
            text="Select User:",
            font=self.font_heading,
        ).pack(side="left")

        # Create StringVar to hold the selected user value
        self.user_var = tk.StringVar()
        
        # Build list of user display names: "ID – Name" format
        user_display_names = [
            f"{user_id:02d} – {self.user_lookup.get(user_id, f'User {user_id}')}"
            for user_id in sorted(self.user_lookup.keys())
        ]

        # User selection dropdown (combobox)
        self.user_combo = ttk.Combobox(
            top_frame,
            textvariable=self.user_var,
            values=user_display_names,
            state="readonly",  # Prevent manual text entry
            width=40,
        )
        self.user_combo.pack(side="left", padx=10)
        # Bind selection event to update display
        self.user_combo.bind("<<ComboboxSelected>>", self._on_user_selected)
        # Select first user by default if users exist
        if user_display_names:
            self.user_combo.current(0)

        # User info display: shows selected user's name, rating count, and average rating
        self.user_info_var = tk.StringVar(value="Select a user to view recommendations.")
        ttk.Label(
            self.root,
            textvariable=self.user_info_var,
            font=self.font_body,
            padding=10,
        ).pack(fill="x")  # Fill entire width

        # Main content area: split into left (recommendations) and right (IMDb details)
        tables_frame = ttk.Frame(self.root, padding=10)
        tables_frame.pack(fill="both", expand=True)  # Fill and expand to use available space

        # Left frame: contains the two recommendation tables
        left_frame = ttk.Frame(tables_frame)
        left_frame.pack(side="left", fill="y", expand=False, padx=5, pady=5)

        # "Likely to Enjoy" table: top 5 recommended movies
        self.like_tree = self._create_treeview(
            left_frame, "Likely to Enjoy", column_widths=(220, 120)
        )
        # Bind selection event to show IMDb details when a movie is clicked
        self.like_tree.bind("<<TreeviewSelect>>", self._on_like_tree_select)
        
        # "Likely to Skip" table: top 5 anti-recommendations
        self.dislike_tree = self._create_treeview(
            left_frame, "Likely to Skip", column_widths=(220, 120)
        )

        # Right frame: IMDb details panel
        right_frame = ttk.Labelframe(tables_frame, text="IMDb Details", padding=10)
        right_frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)

        # Movie poster image label (left side of IMDb panel)
        self.imdb_image_label = ttk.Label(right_frame)
        self.imdb_image_label.pack(side="left", padx=(0, 15))

        # Details frame: contains text information (right side of IMDb panel)
        details_frame = ttk.Frame(right_frame)
        details_frame.pack(side="left", fill="both", expand=True)

        # Movie title label (main title)
        self.imdb_title_label = ttk.Label(details_frame, text="Select a movie", font=self.font_heading)
        self.imdb_title_label.pack(anchor="w")  # Align left

        # Alternate title label (e.g., original title)
        self.imdb_alt_label = ttk.Label(details_frame, text="", font=("Segoe UI", 10, "italic"))
        self.imdb_alt_label.pack(anchor="w")

        # Metadata label: genres and duration
        self.imdb_meta_label = ttk.Label(details_frame, text="")
        self.imdb_meta_label.pack(anchor="w", pady=(6, 0))

        # Rating label: IMDb rating and vote count
        self.imdb_rating_label = ttk.Label(details_frame, text="")
        self.imdb_rating_label.pack(anchor="w")

        # Description text widget: movie plot/description
        self.imdb_desc = tk.Text(
            details_frame,
            wrap="word",  # Wrap at word boundaries
            height=12,
            bg=self.bg_medium_red,  # Match theme background
            fg=self.fg_light,  # Match theme text color
            font=self.font_body,
            insertbackground=self.fg_light,  # Cursor color
            selectbackground=self.accent_red,  # Selection highlight (red)
            selectforeground=self.fg_light,  # Selected text color
            borderwidth=1,
            relief="flat",
        )
        self.imdb_desc.pack(fill="both", expand=True, pady=(10, 0))
        self.imdb_desc.insert("1.0", "IMDb details will appear here.")
        self.imdb_desc.configure(state="disabled")  # Read-only

        # Status bar: displays model training information and metrics
        self.status_var = tk.StringVar(
            value=f"Model trained on {len(self.ratings)} ratings. RMSE={self.metrics['rmse']:.2f}  MAE={self.metrics['mae']:.2f}"
            if not pd.isna(self.metrics["rmse"])
            else "Model trained on full dataset."
        )
        status_label = ttk.Label(
            self.root,
            textvariable=self.status_var,
            relief="sunken",  # Sunken border for status bar appearance
            anchor="w",  # Left-align text
            padding=5,
            font=self.font_small,
        )
        status_label.pack(fill="x", side="bottom")  # Place at bottom, fill width

        # Display initial user data if a user is selected by default
        if self.user_combo.get():
            self.display_selected_user_data()

    def _create_treeview(
        self, parent: ttk.Frame, title: str, column_widths: Tuple[int, int]
    ) -> ttk.Treeview:
        """
        Create a Treeview widget for displaying movie recommendations.
        
        Creates a labeled frame containing a table with two columns (Title and Predicted Score)
        and a vertical scrollbar. Used for both "Likely to Enjoy" and "Likely to Skip" tables.
        
        Args:
            parent: Parent frame to contain the treeview
            title: Label text for the frame (e.g., "Likely to Enjoy")
            column_widths: Tuple of (title_width, score_width) in pixels
            
        Returns:
            The created Treeview widget
        """
        # Create labeled frame container
        frame = ttk.Labelframe(parent, text=title, padding=10)
        frame.pack(fill="both", expand=True, padx=5, pady=5)

        # Create Treeview widget with two columns
        tree = ttk.Treeview(
            frame,
            columns=("title", "score"),  # Column identifiers
            show="headings",  # Show column headers, hide tree structure
            height=12,  # Number of visible rows
        )
        # Configure column headers
        tree.heading("title", text="Title")
        tree.heading("score", text="Predicted")
        # Configure column widths and alignment
        tree.column("title", width=column_widths[0], anchor="w")  # Left-align
        tree.column("score", width=column_widths[1], anchor="center")  # Center-align
        tree.pack(fill="both", expand=True)

        # Add vertical scrollbar
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscroll=scrollbar.set)  # Link scrollbar to treeview
        scrollbar.pack(side="right", fill="y")

        return tree

    def _on_user_selected(self, event: tk.Event) -> None:  # noqa: D401
        """
        Event handler for user selection in the dropdown.
        
        Called when the user selects a different user from the combobox.
        Updates the display to show recommendations for the selected user.
        
        Args:
            event: Tkinter event object (not used but required for binding)
        """
        self.display_selected_user_data()

    def display_selected_user_data(self) -> None:
        """
        Display recommendations and user information for the selected user.
        
        Retrieves recommendations from the model, formats them with metadata,
        populates the recommendation tables, and automatically selects the first
        recommended movie to display its IMDb details.
        """
        # Get selected user from combobox
        selection = self.user_combo.get()
        if not selection:
            messagebox.showinfo("Select User", "Please choose a user from the dropdown.")
            return

        # Parse user ID from selection string (format: "ID – Name")
        user_id = int(selection.split("–")[0].strip())
        user_name = self.user_lookup.get(user_id, f"User {user_id}")

        # Get user's rating history to determine which movies they've already seen
        user_ratings = self.ratings[self.ratings["userId"] == user_id]
        seen_items = user_ratings["movieId"]
        
        # Get recommendations from the model
        recs = self.model.recommend(
            user_id=user_id,
            seen_items=seen_items,
            n_recommendations=5,  # Top 5 movies to recommend
            n_anti_recommendations=5,  # Top 5 movies to avoid
        )
        
        # Format recommendations with movie titles and IMDb IDs
        formatted = format_recommendations_with_metadata(
            recs, self.movie_lookup, self.imdb_lookup
        )

        # Calculate and display user statistics
        avg_rating = user_ratings["rating"].mean() if not user_ratings.empty else 0.0
        self.user_info_var.set(
            f"{user_name} | Ratings: {len(user_ratings)} | Avg rating: {avg_rating:.2f}"
        )

        # Store recommendation data for the "Likely to Enjoy" table
        # This allows us to retrieve full movie data when a row is selected
        self.like_tree_data: Dict[str, Dict[str, object]] = {}
        
        # Populate both recommendation tables
        self._populate_tree(self.like_tree, formatted["recommendations"], self.like_tree_data)
        self._populate_tree(self.dislike_tree, formatted["anti_recommendations"])

        # Automatically select the first recommendation to display IMDb details immediately
        like_items = self.like_tree.get_children()
        if like_items:
            self.like_tree.selection_set(like_items[0])  # Select first item
            self.like_tree.focus(like_items[0])  # Give focus
            self._on_like_tree_select(None)  # Trigger IMDb display

    def _populate_tree(
        self,
        tree: ttk.Treeview,
        rows: List[Dict[str, object]],
        storage: Dict[str, Dict[str, object]] | None = None,
    ) -> None:
        """
        Populate a Treeview widget with movie recommendation data.
        
        Clears existing items and inserts new rows. Optionally stores row data
        in a dictionary for later retrieval (used for "Likely to Enjoy" table).
        
        Args:
            tree: The Treeview widget to populate
            rows: List of dictionaries containing movie data (title, score, imdbId, etc.)
            storage: Optional dictionary to store row data keyed by item_id
        """
        # Clear existing items
        for item in tree.get_children():
            tree.delete(item)
        
        # Insert new rows
        for row in rows:
            item_id = tree.insert(
                "",
                "end",  # Insert at end
                values=(row["title"], f"{row['score']:.2f}"),  # Title and formatted score
            )
            # Store row data if storage dictionary provided (for "Likely to Enjoy" table)
            if storage is not None:
                storage[item_id] = row

    def _on_like_tree_select(self, event: tk.Event) -> None:
        """
        Event handler for movie selection in the "Likely to Enjoy" table.
        
        Called when user clicks on a movie in the recommendations table.
        Retrieves the movie data and updates the IMDb details panel.
        
        Args:
            event: Tkinter event object (may be None if called programmatically)
        """
        selection = self.like_tree.selection()
        if not selection:
            return
        # Get the selected item ID
        item_id = selection[0]
        # Retrieve stored movie data for this item
        row = getattr(self, "like_tree_data", {}).get(item_id)
        if not row:
            return
        # Update IMDb panel with selected movie's data
        self._update_imdb_panel(row)

    def _update_imdb_panel(self, row: Dict[str, object]) -> None:
        """
        Update the IMDb details panel with information for the selected movie.
        
        Checks if IMDb data is cached. If not, displays loading message and fetches
        data asynchronously. If cached, displays immediately.
        
        Args:
            row: Dictionary containing movie data including imdbId and title
        """
        imdb_id = row.get("imdbId")
        if not imdb_id or imdb_id == "N/A":
            self._show_imdb_placeholder("IMDb ID is unavailable for this title.")
            return

        # Check if data is already cached
        data = self.imdb_cache.get(imdb_id)
        if data is None:
            # Show loading state immediately for better UX
            self._show_imdb_loading(row.get("title", "Unknown title"))
            # Fetch data in background thread to avoid blocking UI
            thread = threading.Thread(target=self._fetch_and_display_imdb, args=(imdb_id, row), daemon=True)
            thread.start()
            return

        # Data already cached, display immediately
        self._display_imdb_data(data, row)

    def _show_imdb_loading(self, title: str) -> None:
        """
        Display loading message in the IMDb details panel.
        
        Updates all IMDb panel widgets to show that data is being fetched.
        Provides user feedback during the API call.
        
        Args:
            title: The movie title being loaded
        """
        self.imdb_title_label.config(text=f"Loading IMDb data for: {title}...")
        self.imdb_alt_label.config(text="")
        self.imdb_meta_label.config(text="Please wait, downloading movie information...")
        self.imdb_rating_label.config(text="")
        self.imdb_image_label.config(image="", text="Loading image...")
        self.imdb_image_label.image = None  # Clear image reference
        self.imdb_desc.configure(state="normal")
        self.imdb_desc.delete("1.0", tk.END)
        self.imdb_desc.insert("1.0", "Fetching movie details from IMDb. This may take a few seconds...")
        self.imdb_desc.configure(state="disabled")
        # Force UI update to show loading message immediately
        self.root.update_idletasks()

    def _fetch_and_display_imdb(self, imdb_id: str, row: Dict[str, object]) -> None:
        """
        Fetch IMDb data in a background thread and update UI when complete.
        
        This method runs in a separate thread to avoid blocking the UI.
        Uses root.after() to safely update the UI from the main thread.
        
        Args:
            imdb_id: The IMDb ID (e.g., "tt0314134") to fetch
            row: Dictionary containing movie data (for fallback title)
        """
        try:
            # Fetch data from IMDb API (this may take a few seconds)
            data = fetch_imdb_data(imdb_id)
            # Cache the data for future use
            self.imdb_cache[imdb_id] = data
            # Schedule UI update in main thread (thread-safe)
            self.root.after(0, lambda: self._display_imdb_data(data, row))
        except Exception as exc:  # noqa: BLE001
            # Show error message in main thread (thread-safe)
            self.root.after(0, lambda: self._show_imdb_placeholder(f"Failed to load IMDb data:\n{exc}"))

    def _display_imdb_data(self, data: dict, row: Dict[str, object]) -> None:
        """
        Display IMDb data in the UI panel.
        
        Extracts and formats movie information from the IMDb API response and
        updates all relevant widgets in the IMDb details panel.
        
        Note: Must be called from the main thread (use root.after() from background threads).
        
        Args:
            data: Dictionary containing IMDb API response data
            row: Dictionary containing movie data (for fallback title if needed)
        """
        # Extract data from the "short" section of the API response
        short = data.get("short", {})
        name = short.get("name", row.get("title", "Unknown title"))
        alt_name = short.get("alternateName", "")
        genres = short.get("genre", [])
        duration = short.get("duration", "")
        aggregate = short.get("aggregateRating", {})
        description = short.get("description", "No description available.")

        # Decode HTML entities (e.g., &apos; -> ', &amp; -> &, etc.)
        name = html.unescape(name)
        alt_name = html.unescape(alt_name) if alt_name else ""
        description = html.unescape(description)

        # Update title label
        self.imdb_title_label.config(text=name)
        
        # Update alternate name label (only if different from main title)
        if alt_name and alt_name != name:
            self.imdb_alt_label.config(text=f"Alternate: {alt_name}")
        else:
            self.imdb_alt_label.config(text="")

        # Build metadata string (genres and duration)
        meta_parts = []
        if genres:
            meta_parts.append(f"Genres: {', '.join(genres)}")
        if duration:
            formatted_duration = self._format_duration(duration)  # Format ISO 8601 to readable
            meta_parts.append(f"Duration: {formatted_duration}")
        self.imdb_meta_label.config(text=" | ".join(meta_parts))

        # Update rating label
        if aggregate:
            rating = aggregate.get("ratingValue")
            best = aggregate.get("bestRating", 10)
            votes = aggregate.get("ratingCount", "–")
            self.imdb_rating_label.config(text=f"Rating: {rating}/{best} ({votes} votes)")
        else:
            self.imdb_rating_label.config(text="Rating information not available.")

        # Load and display movie poster image
        poster = load_image_from_url(short.get("image"), max_width=250)
        if poster:
            self.imdb_image_label.config(image=poster, text="")
            self.imdb_image_label.image = poster  # Keep reference to prevent garbage collection
        else:
            self.imdb_image_label.config(image="", text="No image")
            self.imdb_image_label.image = None

        # Update description text
        self.imdb_desc.configure(state="normal")  # Enable editing
        self.imdb_desc.delete("1.0", tk.END)  # Clear existing text
        self.imdb_desc.insert("1.0", description)  # Insert new description
        self.imdb_desc.configure(state="disabled")  # Make read-only again

    def _format_duration(self, duration: str) -> str:
        """
        Format ISO 8601 duration string to human-readable format.
        
        Converts duration strings like "PT1H40M" (ISO 8601 format) to
        readable format like "1 h 40 m". Handles cases with only hours,
        only minutes, or both.
        
        Args:
            duration: ISO 8601 duration string (e.g., "PT1H40M", "PT1H", "PT40M")
            
        Returns:
            Formatted duration string (e.g., "1 h 40 m") or original string if parsing fails
        """
        if not duration:
            return duration
        
        # Parse ISO 8601 duration format: PT1H40M or PT1H or PT40M
        # Extract hours (H) and minutes (M) using regex
        hours_match = re.search(r'(\d+)H', duration)
        minutes_match = re.search(r'(\d+)M', duration)
        
        # Extract numeric values, default to 0 if not found
        hours = int(hours_match.group(1)) if hours_match else 0
        minutes = int(minutes_match.group(1)) if minutes_match else 0
        
        # Build formatted parts
        parts = []
        if hours > 0:
            parts.append(f"{hours} h")
        if minutes > 0:
            parts.append(f"{minutes} m")
        
        # Return formatted string or original if no parts found
        return " ".join(parts) if parts else duration

    def _show_imdb_placeholder(self, message: str) -> None:
        """
        Display a placeholder message in the IMDb details panel.
        
        Used when IMDb data is unavailable or an error occurs.
        Clears all IMDb panel widgets and shows the provided message.
        
        Args:
            message: Error or placeholder message to display
        """
        self.imdb_title_label.config(text="IMDb data unavailable")
        self.imdb_alt_label.config(text="")
        self.imdb_meta_label.config(text="")
        self.imdb_rating_label.config(text="")
        self.imdb_image_label.config(image="", text="No image")
        self.imdb_image_label.image = None
        self.imdb_desc.configure(state="normal")
        self.imdb_desc.delete("1.0", tk.END)
        self.imdb_desc.insert("1.0", message)
        self.imdb_desc.configure(state="disabled")

    def generate_plots(self) -> None:
        """
        Generate and save data analysis plots.
        
        Creates input and output data visualizations and saves them as PNG files
        in the lab3 directory. Shows success or error message to the user.
        """
        try:
            # Create train/test split for plotting
            split = stratified_user_split(self.ratings, test_ratio=0.2, random_state=42)
            # Generate input data plots (ratings distribution, genres, etc.)
            plot_input_data(self.movies, self.ratings, split)
            # Generate output data plots (prediction accuracy, recommendations, etc.)
            plot_output_data(self.model, split.test, self.metrics, self._collect_recs_for_all_users())
            messagebox.showinfo(
                "Plots Saved",
                "Input and output visualizations have been saved in the lab3/ directory.",
            )
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Plot Error", f"Failed to generate plots:\n{exc}")

    def _collect_recs_for_all_users(self) -> Dict[str, Dict[str, List[Dict[str, object]]]]:
        """
        Collect recommendations for all users for plotting purposes.
        
        Generates recommendations for every user in the dataset without
        altering the main GUI's user selection. Used for generating
        comprehensive output visualizations.
        
        Returns:
            Dictionary mapping user names to their recommendations and anti-recommendations
        """
        all_recs: Dict[str, Dict[str, List[Dict[str, object]]]] = {}
        # Iterate through all users
        for user_id in sorted(self.user_lookup.keys()):
            # Get movies this user has already rated
            seen_items = self.ratings.loc[self.ratings["userId"] == user_id, "movieId"]
            # Get recommendations
            recs = self.model.recommend(
                user_id=user_id,
                seen_items=seen_items,
                n_recommendations=5,
                n_anti_recommendations=5,
            )
            # Format with metadata and store by user name
            all_recs[self.user_lookup[user_id]] = format_recommendations_with_metadata(
                recs, self.movie_lookup, self.imdb_lookup
            )
        return all_recs


def main() -> None:
    """
    Main entry point for the GUI application.
    
    Creates the root Tkinter window, initializes the RecommendationGUI,
    and starts the main event loop.
    """
    root = tk.Tk()
    RecommendationGUI(root)
    root.mainloop()  # Start the GUI event loop


if __name__ == "__main__":
    main()

