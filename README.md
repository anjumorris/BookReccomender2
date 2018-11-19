# Book Recommender 

### Project Fletcher

> #### Datasets
> **3000** full e-books from Project Gutenberg - .txt file format
>
> **150** authors
>
> Enhanced dataset of 30,000+ books



> #### Tools
>
> 1. Numpy
> 2. Pandas
> 3. Matplotlib
> 4. Seaborn
> 5. Flask
> 6. Javascript
> 7. HTML
> 8. NLTK
> 9. TextBlob
> 10. PyLDAvis
> 11. Sklearn
> 12. Powerpoint
> 13. Typora
> 14. Jupyter Notebooks



> #### Algorithms
>
> 1. TextBlob Sentiment Analyser
> 2. Porter Stemmer, Snowball Stemmer , Wordnet Lemmatizer
> 3. CountVectorizer, TfidfVectorizer
> 4. Latent Dirichlet Allocation,  Non-negative Matrix factorization
> 5. K Means
> 6. Singular Value Decomposition(SVD) for 2D cluster visualization
> 7. Cosine Similarity



> #### Modules 
>
> 1. **code** - contains notebooks for carrying out NLP 
>
>    1. **process_vectorize.ipynb** 
>
>       Opens all the file list processes them carries out tokenization + lemmatization followed by creating a bag or words out of all the books.
>
>    2. **sentiment_analysis.ipynb**
>
>       Opens all the files one by one and perfoms sentiment analysis on it. 
>
>    3. **dimension_reduction.ipynb**
>
>       Notebook for topic modeling using LDA and NMF.
>
>    4. **featurize_cluster.ipynb**
>
>       This notebook combines the outputs of topic modelling and the sentiment analysis to create features that can be used to cluster.
>
>    5. **similarity.ipynb**
>
>       This notebook finds the cosine similarity between a sample excerpt and the books in our corpus. The logic of this notebook is what is used in the flask app.
>
> 2. **data** - all stored data including dilled/pickled pipelines
>
>    1. **/gutenberg**
>
>       All the original ebook text files in .txt format 
>
>    2. **/samples**
>
>       Excerpts used to test the cosine similarity 
>
>    3. **/vectors**
>
>       stores all the intermediate vectors and pipelines that get used. 
>
>    4. **final_full.csv**
>
>       final corpus used to cluster / perform similarity analysis
>
>    5. **sentiment_full.csv**
>
>       final sentiment analysis of all books. used 
>
> 3. **images**
>
>    1. stores images for making the presentation. also some csv files for using to make excel charts.
>
> 4. **templates & static** 
>
>    1. static/index.html - landing page for flask app
>
> 5. **app.py**
>
>    1. Flask app for giving similar book suggestion
>    2. Input_sample.txt -> temp file where user excerpt from the app is stored 
>
> 6. **docs** - documents 
>
>    1. Proposal - proposal_project_fletcher.pdf
>    2. Presentation -PPT_book_recommender .pdf
>    3. Summary - Summary_Book_recommender.pdf
>



> #### How to run ?
>
> 1. Run the following code notebooks - featurize_cluster.ipynb , similarity.ipynb
> 2. Run flask app -> python app.py

