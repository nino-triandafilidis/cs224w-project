# CS 224W Course Project (Fall 2026): Project 1 excerpt

> **Source:** "CS 224W Course Project 2026" (Google Doc), https://docs.google.com/document/d/1LrCJJ54tbAHgMbTDT7nVTyllTPymjkgS
>
> **Pulled:** 2026-10-18, from the .docx export.
>
> **Scope:** excerpt for Project type 1 (Real-world applications of GNNs). Text is copied verbatim; Project 2 and 3 content is dropped and each omission is marked inline. Headings were renumbered to Markdown levels and the table of contents rebuilt for the kept sections; "1.1 Example graphs & datasets" is plain text in the source and is made a heading here. Highlighting and underlining from the doc are dropped. If anything here disagrees with the live doc, the live doc wins.

**Table of Contents**

- [0. Overview](#0-overview)
- [1. Real-world applications of GNNs](#1-real-world-applications-of-gnns)
  - [1.1 Example graphs & datasets](#11-example-graphs--datasets)
  - [1.2 Examples of good projects](#12-examples-of-good-projects)
  - [1.3 Project Proposal \[13.33%\]](#13-project-proposal-1333)
  - [1.4 Project Milestone \[6.67%\]](#14-project-milestone-667)
  - [1.5 Project Report \[80%\]](#15-project-report-80)
- *[2. Tutorial on PyG functionality — omitted (Project 2)]*
- *[3. Implementation of cutting-edge research — omitted (Project 3)]*
- [4. Project logistics](#4-project-logistics)
  - [4.1 What to submit](#41-what-to-submit)
  - [4.2 Project Feedback and Mentor Assignments](#42-project-feedback-and-mentor-assignments)
  - [4.3 Writing blog posts on Medium](#43-writing-blog-posts-on-medium)
  - [4.4 Instructions on Google Colab](#44-instructions-on-google-colab)
  - [4.5 Compute resources](#45-compute-resources)
  - [4.6 Project sharing policy](#46-project-sharing-policy)
- [5. Extra credit for approved PRs](#5-extra-credit-for-approved-prs)
- [6. Tips on writing blog posts](#6-tips-on-writing-blog-posts)

## 0. Overview

The goal of this year’s CS224W course project is to help you create long-lasting resources for both your technical profiles and the graph machine learning community at large. To this end, we are inviting three types of projects:

1)  **[Real-world applications of GNNs](#1-real-world-applications-of-gnns) \[blog post + Google Colab\]**: in this project, you will identify a specific use case (e.g., predicting drug interactions, fraud detection) and demonstrate how GNNs + PyG can be used to solve this problem.

2)  **Tutorial on PyG functionality \[blog post + Google Colab\]**: in this project, you will develop a tutorial that explains how to use existing PyG features and functionality, such as methods for GNN explainability or graph sampling.

3)  **[Implementation of cutting-edge research](https://docs.google.com/document/d/1S0xG8l1JMs488y8MtSvba-ZAQyv9isUGov0qusoKppg/edit#heading=h.ym5e9vt4dbsf) \[blog post + PR\]**: in this project, you will identify a recent research paper in graph ML that develops interesting methods and implement the work as a pull request (PR) to the [contrib](https://pytorch-geometric.readthedocs.io/en/latest/modules/contrib.html) package in PyG.

Graph ML is one of the hottest emerging topics in machine learning, with wide applicability across domains and industries. We hope your project will first serve as an opportunity for you to dive deep into graph ML algorithms and get first-hand experience implementing them. Furthermore, the outputs of your projects will become great resources for the broader graph ML community, particularly for newcomers who want to learn about where GNNs can be applied and how to use them.

Through your projects, you will gain experience using [PyG](https://pytorch-geometric.readthedocs.io/en/latest/) (PyTorch Geometric), the most popular graph deep learning framework built on top of PyTorch. PyG is suitable for quickly implementing GNN models, with [abundant graph models](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#) already implemented and [datasets](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html) ready to load. Furthermore, [GraphGym](https://pytorch-geometric.readthedocs.io/en/latest/advanced/graphgym.html), available as part of PyG, allows a training/evaluation pipeline to be built in a few lines of code.

Your projects are also an opportunity to demonstrate your knowledge of graph ML to a wide audience. We will publish all blog posts on [our course’s Medium page](https://medium.com/stanford-cs224w) and encourage you to share your hard work on your websites / social media. We will also publish the best blog posts at [PyG.org](https://www.pyg.org/) and hope to integrate your high-quality PRs into the main PyG library!

We recommend project groups of 3 students, but groups of 2 are also allowed. Solo projects are discouraged; please see Ed. **Project proposals are due on Tuesday, October 20, 2026; the project milestone is due Tuesday, November 10, 2026; and final project reports are due on Thursday, December 10, 2026.** *[Dates updated from the course schedule, web.stanford.edu/class/cs224w; the source doc still showed 2025 dates here.]* Please see [Project Logistics](#4-project-logistics) for details.

## 1. Real-world applications of GNNs

*Output: blog post, Google Colab*

In this project, you will identify a specific use case and demonstrate how GNNs and PyG can be used to solve this problem. We welcome use cases across domains (e.g., biological, social, financial), but we want to see use cases that are motivated by real-world problems, e.g., an application that a company would actually use. Given a use case, we expect you to identify an appropriate *public* dataset, formulate the use case as a clear graph ML problem, and demonstrate how GNNs can be used to solve this problem.

Note that it is not enough to simply *apply* an existing GNN model to a new dataset. We expect to see some form of **novelty** (e.g. developing a new GNN method for your specific problem, improving on existing methods in a non-trivial way) or **comprehensive analyses** (e.g. ablation studies, comparison between multiple model architectures) for the project.

### 1.1 Example graphs & datasets

Below, we list several real-world application domains, specifying how graphs and tasks are defined in these domains. We also provide representative graph ML models and public datasets that you may use, such as from [Open Graph Benchmark](https://ogb.stanford.edu/) (OGB)**\*** and the datasets available in [PyG](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html). These are just examples; you are free to explore other application domains, models, and public datasets yourself.

**\***Note that some OGB datasets are quite large and require a decent amount of computational resources to handle. We recommend using small-scale OGB datasets that are more tractable.

***Recommender systems***

**Graphs**:

- Nodes: Users, items

- Edges: User-item interactions

**Tasks**:

- Predicting the edge ratings. Metric: RMSE

- Predicting edge existence. Metric: Recall@K

**Example model(s):** [LightGCN](https://arxiv.org/abs/2002.02126)

**Public datasets**: [MovieLens](https://grouplens.org/datasets/movielens/), [Recsys repository](https://cseweb.ucsd.edu/~jmcauley/datasets.html)

***Product co-purchasing graph***

**Graphs:**

- Nodes: Products

- Edges: Product co-purchasing relations

**Tasks:**

- Predicting product categories. Metric: Classification accuracy

**Example model(s):** See the [leaderboard](https://ogb.stanford.edu/docs/leader_nodeprop/#ogbn-products)

**Datasets:** [ogbn-products](https://ogb.stanford.edu/docs/nodeprop/#ogbn-products)

**Note**: This is a medium-scale OGB dataset that requires more computational resources than the other datasets.

***Fraud detection in transaction graphs***

**Graphs:**

- Nodes: Financial users (customers, banks)

- Edges: Transaction (money and amount sent)

**Tasks:**

- Edge classification - predict which edges are fraudulent. Metric: Hits@50

**Example model(s):** See [https://github.com/safe-graph/graph-fraud-detection-papers](https://github.com/safe-graph/graph-fraud-detection-papers)

**Datasets:** [Bitcoin Fraud Dataset (only use labeled data!)](https://www.kaggle.com/ellipticco/elliptic-data-set)

***Friend recommendation***

**Graphs:** social network

- Nodes: users

- Edges: potentially heterogeneous -- friend, follow, reply to, message, like, etc.

**Tasks:**

- Recommending/ranking new friends for users. Metrics: Hits@K, NDCG@K, MRR

**Example model(s):** GraFRank ([paper](http://nshah.net/publications/GrafRank.WWW.21.pdf), [GitHub](https://github.com/aravindsankar28/GraFRank))

**Datasets:** [Facebook](http://snap.stanford.edu/data/ego-Facebook.html), [Google+](http://snap.stanford.edu/data/ego-Gplus.html), [Twitter](http://snap.stanford.edu/data/ego-Twitter.html)

***Paper citation graphs***

**Graphs:**

- Nodes: Papers

- Edges: Paper citations

**Tasks:**

- Predicting subject areas of papers. Metric: Classification accuracy

**Example model(s):** See the [leaderboard](https://ogb.stanford.edu/docs/leader_nodeprop/#ogbn-arxiv)

**Datasets:** [ogbn-arxiv](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv)

***Author collaboration networks***

**Graphs:**

- Nodes: Authors

- Edges: Author Collaboration

**Tasks:**

- Predicting future author collaboration. Metric: Hits@50

**Example model(s):** See the [leaderboard](https://ogb.stanford.edu/docs/leader_linkprop/#ogbl-collab)

**Datasets:** [ogbl-collab](https://ogb.stanford.edu/docs/linkprop/#ogbl-collab)

***Heterogeneous academic graph***

**Graphs:**

- Nodes: Papers, authors, institutions, fields of study

- Edges: an author is “affiliated with” an institution, an author “writes” a paper, a paper “cites” a paper, and a paper “has a topic of” a field of study

**Tasks:**

- Predicting paper publication venues. Metric: Classification accuracy

**Example model(s):** See the [leaderboard](https://ogb.stanford.edu/docs/leader_nodeprop/#ogbn-mag)

**Datasets:** [ogbn-mag](https://ogb.stanford.edu/docs/nodeprop/#ogbn-mag)

**Note**: This is a medium-scale OGB dataset that requires more computational resources than the other datasets.

***Knowledge graphs***

**Graphs:**

- Nodes: Entities

- Edges: Knowledge triples

**Tasks:**

- Predicting missing triples. Metric: Mean Reciprocal Rank (MRR)

**Example model(s):** [TransE](https://proceedings.neurips.cc/paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html), [DistMult](https://arxiv.org/abs/1412.6575), [ComplEx](https://arxiv.org/abs/1606.06357), [RotatE](https://arxiv.org/abs/1902.10197)

**Datasets:** [FB15k-237](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding/tree/master/data/FB15k-237), [WN18RR](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding/tree/master/data/wn18rr)

***Molecule classification***

**Graphs:**

- Nodes: Atoms

- Edges: Bonds

**Tasks:**

- Predicting properties of molecules. Metric: ROC-AUC

**Example model(s):** See the [leaderboard](https://ogb.stanford.edu/docs/leader_graphprop/#ogbg-molhiv)

**Datasets:** [ogbl-molhiv](https://ogb.stanford.edu/docs/graphprop/#ogbg-mol)

***Protein-protein interaction networks***

**Graphs:**

- Nodes: Gene nodes

- Edges: Interaction between gene nodes

**Tasks:**

- Node classification - protein function prediction. Metric: Classification accuracy

**Example model(s):** See methods on [OGB node classification leaderboard](https://ogb.stanford.edu/docs/leader_nodeprop/)

**Datasets:** [https://snap.stanford.edu/biodata/datasets/10013/10013-PPT-Ohmnet.html](https://snap.stanford.edu/biodata/datasets/10013/10013-PPT-Ohmnet.html)

***Drug-drug interaction networks***

**Graphs:**

- Nodes: FDA-approved or experimental drug

- Edges: interactions between drugs (joint effect of taking the two drugs together)

**Tasks:**

- predict drug-drug interactions - Metric: Hits@K

**Example model(s):** See the [leaderboard](https://ogb.stanford.edu/docs/leader_nodeprop/#ogbn-products)

**Datasets:** [ogbl-ddi](https://ogb.stanford.edu/docs/leader_linkprop/#ogbl-ddi)

### 1.2 Examples of good projects

Here we provide examples of good projects from the last two years’ students, of applications of GNNs to impactful real-world use cases. You can find all of the blog posts from our course here: [https://medium.com/stanford-cs224w](https://medium.com/stanford-cs224w).

- [Stock Market Forecasting with Differential Graph Transformer](https://medium.com/stanford-cs224w/stock-market-forecasting-with-differential-graph-transformer-62d095ebc821) by Xiang Li and Farzad Pourbabaee

- [Tracing Historical Political Leaning in Newspapers with GNNs](https://medium.com/stanford-cs224w/tracing-historical-political-leaning-in-newspapers-with-gnns-948b253dee9d) by Amy Guan, Poonam Sahoo, Karsen Wahal

- [Predicting Water Quality in Unmonitored Networks with RGCNs](https://medium.com/stanford-cs224w/predicting-water-quality-in-unmonitored-networks-with-rgcns-c72fdd177f0d) by Tim Chen, Zhenyu Zhang, and Shuojia Fu

- [Code Similarity Using Graph Neural Networks](https://medium.com/stanford-cs224w/code-similarity-using-graph-neural-networks-1e58aa21bd92) by Abhay Singhal, Suhas Chundi, and Patrick Donohue

- [graphE2G: Linking Genes and Their Regulatory Elements Using Graph Neural Network](https://medium.com/stanford-cs224w-fall-2023-course-projects/graphe2g-linking-genes-and-their-regulatory-elements-using-graph-neural-networks-a307c29341bf) by Valeh Valiollah Pour Amiri, Tony Kyle Zeng, Garyk Brixi

- [Spread No More: Twitter Fake News Detection with GNN](https://medium.com/stanford-cs224w/spread-no-more-twitter-fake-news-detection-with-gnn-b90ec6c41828) by Li Tian, Sherry Wu, and Yifei Zheng

- [Learning Complex Dynamics of Particle Interactions through a Social Dynamics Lens](https://medium.com/stanford-cs224w/learning-complex-dynamics-of-particle-interactions-through-a-social-dynamics-lens-f24f05a9ea8) by Nitish Gudapati, Shashvat Jayakrishnan, and Shreya Agrawal

- [Automated Theorem Proving with Graph Neural Networks](https://medium.com/stanford-cs224w/automated-theorem-proving-with-graph-neural-networks-49c091024f81) by Dan Jenson, Julian Cooper, and Daniel Huang

- [Augmenting Your Notes Using Graph Neural Networks](https://medium.com/stanford-cs224w/augmenting-your-notes-using-graph-neural-networks-e61f0898033a) by Arjun Karanam and Michael Elabd

- [Re-Mix and Match!](https://medium.com/stanford-cs224w/re-mix-and-match-829b187139b0) by Jay Yu, Laura Wu, and Jiacheng Hu

- [Fraudulent Activity Recognition in Graph Networks](https://medium.com/stanford-cs224w/fraudulent-activity-recognition-in-graph-networks-f60dc724feea) by Ansh Khurana, Aman Kansal, and Soumya Chatterjee

- [Traffic Signal Control Using Decentralized MARL via Graph Modeling](https://medium.com/stanford-cs224w/traffic-signal-control-using-decentralized-marl-via-graph-modeling-436c9666630d) by Kevin Chen, Kelvin Christian, Haoyu Wang

- [Spotify Track Neural Recommender System](https://medium.com/stanford-cs224w/spotify-track-neural-recommender-system-51d266e31e16) by Eva Batelaan, Thomas Brink, and Benjamin Wittenbrink

- [RingSAGE: Cycle-Aware Molecular Regression with Virtual Nodes](https://medium.com/stanford-cs224w/gnns-for-molecule-regression-a704d61de396) by Pranava Singhal, Herumb Shandilya and Harshvardhan Agarwal

### 1.3 Project Proposal \[13.33%\]

The project proposal is **13.33%** of the project grade and **4%** of your total grade.

For this project type, we expect you to cover the following in your proposal:

- Application domain

  - Which dataset(s) are you planning to use?

    - Is the dataset public? Do you plan to make it public before submitting the project? \[*Note: this is a strict requirement\]*

  - Describe the dataset/task/metric. Cite as relevant, especially the data.

  - Why did you choose the dataset?

- Graph ML techniques that you want to apply

  - Which graph ML model are you planning to use? (Be sure to cite!)

  - Describe the model(s) (try using figures and equations).

  - Why is/are the model(s) appropriate for the dataset(s) you have chosen?

### 1.4 Project Milestone \[6.67%\]

The project milestone is **6.67%** of the project grade and **2%** of your total grade. *It is graded on a credit/no credit basis.*

The purpose of the milestone is to encourage you to get started on the project early, and for the TAs to have something more substantial to give feedback on.

For this project type, possible deliverables for the milestone include:

- Code for

  - Processing the dataset

  - Training/evaluating the model

  - Any other programs required by your project

- Report draft

  - Problem description/motivation

  - Dataset description and processing

  - Model design and metrics

Note that you do not need to submit everything listed in the above bullet points. As long as you show that you have done substantial work by the submission deadline, you will receive full credit.

### 1.5 Project Report \[80%\]

This final submission is **80%** of the project grade and **24%** of your total grade.

Deliverables for the final submission and their weights are:

- Blog post (50 points)

  - Motivation & explanation of data/task (10 points)

  - Appropriateness & explanation of model(s) (10 points)

  - Insights + results (10 points)

  - Figures (10 points)

  - Code snippets (10 points)

- Colab (25 points)

  - Code: correctness, design (15 points)

  - Documentation: class/function descriptions, comments in code (10 points)

- Medium account name (5 points)

*[Section 2, Tutorial on PyG functionality (2.1–2.4), omitted: Project 2 only]*

*[Section 3, Implementation of cutting-edge research (3.1–3.4), omitted: Project 3 only]*

## 4. Project logistics

We recommend project groups of 3 students, particularly for Project 3, but groups of 2 are also allowed. Solo projects are discouraged; please see Ed. The project is worth 30% of your course grade.

### 4.1 What to submit

**What to submit for project proposal \[13.33% of *project* grade, 4% of *total* grade\]**

*Due October 20, 2026, 11:59pm PT*

- Who is in your group

- Please refer to the Project Proposal subsection under your project type for instructions on what to include in your proposal

- **Format:** please use the [NeurIPS’22 style file](https://nips.cc/Conferences/2022/PaperInformation/StyleFiles) and aim for 2 pages without references. The abstract is not needed.

**What to submit for project milestone \[6.67% of *project* grade, 2% of *total* grade\]**

*Due November 10, 2026, 11:59pm PT*

- Please refer to the Project Milestone subsection under your project type for instructions on what to include in your proposal

**What to submit for final project \[80% of *project* grade, 24% of *total* grade\]**

*Due December 10, 2026, 11:59pm PT*

- Private draft link to blog post (see instructions below)

- Name of Medium account

- Google Colab (Projects 1+2) or PR to PyG’s [contrib](https://pytorch-geometric.readthedocs.io/en/latest/modules/contrib.html) (Project 3)

### 4.2 Project Feedback and Mentor Assignments

Each group will be assigned a course assistant as a mentor for their project whose purpose is to provide feedback for your project as you progress through the quarter. Meeting with your assigned mentor is not required, but it is recommended that you meet with them at least once during the quarter to make sure you are headed in the right direction.

Mentor assignments will begin after the due date for submitting project proposals (October 20, 2026).

### 4.3 Writing blog posts on Medium

For all three project types, a final product of your project will be your blog post. For this course, you will be writing and we will be releasing your blog posts on [Medium](https://medium.com/). Writing a blog post in Medium is easy and intuitive; see step-by-step instructions below. We also provide [tips](#6-tips-on-writing-blog-posts) on how to write an engaging blog post.

1.  [Sign up / sign in on Medium.](https://medium.com/m/signin)

2.  For each group, one member should set up a draft

    1.  To start a new draft, go [here](https://medium.com/new-story).

    2.  To restart from the existing draft, go [here](https://medium.com/me/stories/drafts).

3.  Editing the draft

    1.  Only one member (who owns the draft) can directly edit the blog post.

    2.  If you want to write the blog together with your members, we suggest you work on the Google doc first and then copy-paste the eventual texts/images to the Medium draft.

4.  Submitting your draft

    1.  Your project submission will include the **draft link** to your blog post. To get the link: on the editing page, click the “...” button (located right next to the “publish”), then click “Share draft link.”

    2.  Please also share your account name with us.

    3.  **Note: Do NOT publish your blog without our review.**

5.  After the course

    1.  Publishing on the [CS224W Medium page](https://medium.com/stanford-cs224w)

        1.  After the course, you will have a chance to incorporate final feedback into your blog post, then we will work on publishing it on our course’s Medium page.

        2.  To do this, we will add you as a writer to the [stanford-cs224w publication page](https://medium.com/stanford-cs224w).

        3.  Once you are added as our writer, you can click the “...” button and further click “Add to publication”.

    2.  We will also work with the PyG team to select blog posts of interest to publish at [PyG.org](http://pyg.org) and on our social media.

    3.  You will get grading regardless of whether your blogs are published or not.

### 4.4 Instructions on Google Colab

Your Google Colab should include:

- A high-level summary of what the code is about and what the task is

- All the code to reproduce your results in the blog posts (including data preprocessing, model definition, and train/evaluation pipeline).

- Detailed comments of what each cell does.

[See here](https://pytorch-geometric.readthedocs.io/en/latest/notes/colabs.html) for examples of good Google Colabs.

### 4.5 Compute resources

**Google cloud [retrieval link](https://gcp.secure.force.com/GCPEDU?cid=8QwPTjrNEwb5YpbchCZKXORhonc5ULc3IOse7VSaVdS%2FJziUstbuc1kp9J1WGn3t/).**

1.  You will be asked for a name and email address, which needs to match your school domain (either @stanford.edu or @cs.stanford.edu).

2.  A confirmation email will be sent to you with a coupon code.

3.  When you redeem your coupon, make sure you are logged into a **personal Google account** (unlike in step 1).

4.  You can only request ONE code per unique email address.

5.  To create a VM, you can take a look at the guidelines [here](https://medium.com/google-cloud/how-to-run-deep-learning-models-on-google-cloud-platform-in-6-steps-4950a57acfa5) (starting from **Step 3**) and [here](https://github.com/yanxi0830/gcloud-tutorial) (see **Set Up Google Cloud VM Image**). If you need GPU for your project, you will need to request GPU quota -- see the instructions under **Request GPU Quota** in the second link. Please do this soon, since it can take a few days for your request to be approved.

**Once you have the GCP credits, follow [this tutorial](https://medium.com/@senthilnathangautham/colab-gcp-compute-how-to-link-them-together-98747e8d940e) to link them to your Colab notebook.**

NOTE: Google also provides [\$300 in credits](https://cloud.google.com/free) for new customers, so definitely check that out too if it applies to you! Please let us know if you have any issues with redeeming credits or finding sufficient compute resources for your projects.

### 4.6 Project sharing policy

Policy on sharing a final project with another course:

- You should make sure that you follow all the guidelines and requirements for the CS224W project (in addition to the requirements of the other class). So, if you’d like to combine your CS224W project with a class X but class X’s policies don’t allow for it, you cannot do it.

- You cannot turn in an identical project for both classes, but you can share common infrastructure/code base/datasets across the two classes.

- In your milestone and final report, clearly indicate which part of the project is done for CS224W and which part is done for a class other than CS224W. For shared projects, we also require that you submit the final report from the class you’re sharing the project with.

## 5. Extra credit for approved PRs

- You will also be considered for up to an extra 1-3% of your course grade if you create *approved* pull requests to [OGB](https://github.com/snap-stanford/ogb), [PyG](https://github.com/pyg-team/pytorch_geometric), or [GraphGym](https://github.com/pyg-team/pytorch_geometric/tree/master/torch_geometric/graphgym) that add useful functionalities or fix bugs, depending on the significance of the contribution.

- In the spirit of open-source development, you are highly encouraged to participate in improving these resources as you work with them throughout the course and in your final projects. For example, for each project, we will give extra credit for the following:

  - Project 1: contribute your final project to the OGB leaderboard together with open-source code using PyG. This is particularly exciting if your model is able to achieve SoTA performance!

  - Project 1 or 2: write a blog post that we choose to feature on the PyG website

  - *[Project 3 extra-credit bullet omitted]*

- Also, we encourage students doing any project to:

  - Find bugs and create issues

  - Propose missing functionality and implement it

  - Contribute useful modules to the GraphGym pipeline

- We highly recommend consulting the TAs before contributions so that we can confirm the details of incorporating the contribution and granting the extra grade.

## 6. Tips on writing blog posts

First, please read [this article](https://towardsdatascience.com/questions-96667b06af5) carefully to learn about how to write machine learning blog posts. For this course project, please also follow the instructions below.

**In the blog posts, you should include the following:**

- At the beginning of your blog post, include “By XXX, YYY, ZZZ as part of the Stanford CS224W course project.”, where XXX, YYY, ZZZ are the names of the team members.

- Wherever applicable:

  - The domain(s) that you are applying graph ML to.

  - Dataset descriptions (source, pre-processing etc).

  - Results that you obtain using the model on the dataset

- Paper references as \[1\], \[2\], etc., see this [blog post](https://towardsdatascience.com/geometric-foundations-of-deep-learning-94cdd45b451d) as an example.

- Step-by-step explanation of graph ML techniques you are using

  - You can assume the following for the readers.

    - Readers are familiar with machine learning (e.g., [CS229](https://cs229.stanford.edu/)) and deep learning (e.g., ​​[CS230](https://cs230.stanford.edu/)) concepts. You do not need to explain them in detail.

    - Readers are familiar with PyTorch.

    - Readers are not familiar with graph ML.

  - Some code snippets of how you used PyG/PyTorch to implement the techniques

- **Visualizations** that would make the blog posts intriguing to read.

  - Gifs \> Images \> Text to show your methods and results.

  - Try to use videos, images, flow charts as much as possible.

  - The more visualization, the better. Reading text-occupied blogs is often painful.

  - Provide image credits if you are adopting figures from other places (we encourage you to make your own figures).

- Link to your Google Colab that can be used to reproduce your results.

- Avoid criticizing research / research orgs. You are here to showcase your work, not to write opinion pieces.

**A good blog post should**

- be fun to read with many figures and visualizations.

- be easy to follow even for graph ML novice.

- clearly convey the potential of graph ML.

- contain a good amount of PyG code snippets, with explanations, to understand how PyG is used in the project.

- be around 10 minutes to read (although this is not a hard constraint).

**Examples of good blog posts:**

- [http://peterbloem.nl/blog/transformers](http://peterbloem.nl/blog/transformers)

- [https://tkipf.github.io/graph-convolutional-networks/](https://tkipf.github.io/graph-convolutional-networks/)

- [https://towardsdatascience.com/graph-neural-networks-as-neural-diffusion-pdes-8571b8c0c774](https://towardsdatascience.com/graph-neural-networks-as-neural-diffusion-pdes-8571b8c0c774)

- [https://blog.twitter.com/engineering/en_us/topics/insights/2021/temporal-graph-networks](https://blog.twitter.com/engineering/en_us/topics/insights/2021/temporal-graph-networks)
