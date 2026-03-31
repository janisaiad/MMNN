# Neural Network Architecture Beyond Width and Depth

## Paper 1 — NestNet (NeurIPS 2022); metadata, abstract, forum UI

(/pdf?id=36-xl1wdyu)
Shijun Zhang (/profile?id=~Shijun_Zhang1), Zuowei Shen (/profile?
id=~Zuowei_Shen1), Haizhao Yang (/profile?id=~Haizhao_Yang1)
Published: 01 Nov 2022, Last Modified: 15 Jan 2023 NeurIPS 2022 Accept Readers: 
Everyone

Show Bibtex Show Revisions (/revisions?id=36-xl1wdyu)
Keywords: Neural Network Approximation, Nested Architecture, Parameter Sharing, Function Composition
Abstract: This paper proposes a new neural network architecture by introducing an additional dimension called height
beyond width and depth. Neural network architectures with height, width, and depth as hyper-parameters are called
three-dimensional architectures. It is shown that neural networks with three-dimensional architectures are significantly
more expressive than the ones with two-dimensional architectures (those with only width and depth as hyper
parameters), e.g., standard fully connected networks. The new network architecture is constructed recursively via a nested
structure, and hence we call a network with the new architecture nested network (NestNet). A NestNet of height  is built
with each hidden neuron activated by a NestNet of height 
≤s−1
. When 
s =1
s
, a NestNet degenerates to a standard
network with a two-dimensional architecture. It is proved by construction that height- ReLU NestNets with 
parameters can approximate -Lipschitz continuous functions on  
O(n)
1
approximation error of standard ReLU networks with  
extended to generic continuous functions on  
[0, 1]d
[0, 1]d
O(n)
with an error 
parameters is 
O(n−2/d)
s
O(n−(s+1)/d)
, while the optimal
. Furthermore, such a result is
with the approximation error characterized by the modulus of
continuity. Finally, we use numerical experimentation to show the advantages of the super-approximation power of ReLU
NestNets.
Supplementary Material:   

pdf (/attachment?id=36-xl1wdyu&name=supplementary_material)
Add
Public Comment
Reply Type: Author:
all
Hidden From:
[–]
nobody
everybody
Visible To:
all readers
Virtual Presentation by Paper9259 Authors
NeurIPS 2022 Conference Paper9259 Authors
26 Oct 2022, 12:51 NeurIPS 2022 Conference Paper9259 Authors Virtual
Presentation
Readers: Program Chairs, Paper9259 Authors Show Revisions
(/revisions?id=ct_culVYCV)
Virtual Presentation: no
15 Replies
Add
[–]
In-person Presentation by Paper9259 Authors
NeurIPS 2022 Conference Paper9259 Authors
12 Oct 2022, 01:24 NeurIPS 2022 Conference Paper9259 Authors In-person
Presentation
Readers: Program Chairs, Paper9259 Authors Show Revisions
(/revisions?id=hI_5NUwg6z)
In-person Presentation: yes
Public Comment
Add
Public Comment
1/8
https://openreview.net/forum?id=36-xl1wdyu&referrer=%5Bthe profile of Shijun Zhang%5D(%2Fprofile%3Fid%3D~Shijun_Zhang1)
15 Sept 2022, 04:53 NeurIPS 2022 Conference Paper9259 Decision Readers: 
Everyone Show Revisions (/revisions?id=p5jR0jiUJZz)

## Decision — NeurIPS 2022

Paper Decision
NeurIPS 2022 Conference Program Chairs
Decision: Accept
Add Public Comment
[–]

27 Aug 2022, 18:01 NeurIPS 2022 Conference Paper9259 Meta Review Readers:
 Everyone Show Revisions (/revisions?id=Yr_4pDxVHV)

## Meta-review (Area Chair 6qvf)

Meta Review of Paper9259 by Area Chair 6qvf
NeurIPS 2022 Conference Paper9259 Area Chair 6qvf
Recommendation: Accept
Confidence: Certain
Metareview:
The authors propose a new architecture which has superior approximation rates for a given number of parameters;
this is a very interesting notion that is shown on a simple example to be quite effective. The reviewers are supportive
of the paper with their main concerns being the added computational cost and the lack of any examples on real
world data sets, even small ones such as MNIST. The authors should include a small experimental section showing
the text accuracy for MNIST or similar, along with the computational time for both training and applying the
network.
Award: Yes
Add Public Comment
[–]

09 Aug 2022, 15:49 NeurIPS 2022 Conference Paper9259 Reviewers Author Rebuttal
Acknowledgement Readers: Program Chairs, Paper9259 Senior Area Chairs,
Paper9259 Area Chairs, Paper9259 Authors, Paper9259 Reviewer zWh2 Show
Revisions (/revisions?id=vmp3Tfy295)

## Author rebuttal acknowledgement

Author Rebuttal Acknowledgement by Paper9259 Reviewer zWh2
NeurIPS 2022 Conference Paper9259 Reviewer zWh2
Author Rebuttal Acknowledgement: Yes
Add Public Comment
[–]
12 Jul 2022, 13:40 NeurIPS 2022 Conference Paper9259 Official Review Readers:
 Everyone Show Revisions (/revisions?id=cb_lpt_xZiB)

## Official review — Reviewer 7MPy

Official Review of Paper9259 by Reviewer 7MPy
NeurIPS 2022 Conference Paper9259 Reviewer 7MPy
Summary:
The authors propose a novel neural network architecture that adds a new "height" dimension through a recursive
construction. The authors show that their proposed network has better asymptotic error than standard ReLU
networks when accounting for similar  number of parameters, and on a class on Lipschitz continuous function
in .
Strengths And Weaknesses:
Originality:
(+) To the best of my knowledge this recursive "height" network architecture is novel.
Quality:
I looked at the proof but am unfamiliar field.
Clarity:
(+) The paper is well written and easy to understand, even for someone not up to date on theory of NN.
[–]

O(n)
[0,1]d
3/28/26, 9:57 PM Neural Network Architecture Beyond Width and Depth | OpenReview
https://openreview.net/forum?id=36-xl1wdyu&referrer=%5Bthe profile of Shijun Zhang%5D(%2Fprofile%3Fid%3D~Shijun_Zhang1) 2/8
Significance:
(+) To my limited understanding the approximation error that these proposed networks achieve is meaningfully
better than standard NNs.
Questions:
To be clear I come from an applied ML perspective, which colors my questions. My main question is understanding
the value of this work. Is the value:
A) Primarily theoretical in understanding a new class of NNs which can achieve better approximation error?
B) Also applied in that the authors expect their proposed class of NNs to be used on modern ML tasks?
If A), then I think the work is valuable. If B), then I am concerned about the addition of hyper-parameters and the
lack of experiments on any real task.
Limitations:
The authors describe theoretical limitations in their analysis, which I think are reasonable and am unconcerned
about.
Ethics Flag: No
Soundness: 3 good
Presentation: 3 good
Contribution: 3 good
Rating: 6: Weak Accept: Technically solid, moderate-to-high impact paper, with no major concerns with respect to
evaluation, resources, reproducibility, ethical considerations.
Confidence: 2: You are willing to defend your assessment, but it is quite likely that you did not understand the
central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were
not carefully checked.
Code Of Conduct: Yes
Add Public Comment
02 Aug 2022, 10:17 NeurIPS 2022 Conference Paper9259 Official
Comment Readers:  Everyone Show Revisions (/revisions?
id=rBqTj7MUVLe)

## Author response — Reviewer 7MPy

Response to Reviewer 7MPy
NeurIPS 2022 Conference Paper9259 Authors
Comment:
We thank the reviewer for the positive evaluation of our paper. The primary goal of our paper is to design a
new NN architecture by introducing one more dimension height for the purpose of theoretically achieving a
better approximation error than standard NNs. We conduct a simple experiment as a proof of concept for our
new NNs and we believe our new NNs can be further developed and applied to real ML tasks. However, tuning
hyper-parameters for applying our new NNs to real ML tasks would require significant work, and hence it is
left for future research.
Add Public Comment
[–]

12 Jul 2022, 03:43 NeurIPS 2022 Conference Paper9259 Official Review Readers:
 Everyone Show Revisions (/revisions?id=efkq0obhlCt)

## Official review — Reviewer zWh2

Official Review of Paper9259 by Reviewer zWh2
NeurIPS 2022 Conference Paper9259 Reviewer zWh2
Summary:
The paper proposes a new family of neural architectures which is given not just by width and depth but also by
'height', a newly introduced dimension. Such models are allowed to have activations that are themselves realized by
other networks, resulting in an implicit parameter sharing scheme. Results on approximation theory are shown for
such family of models, where they have better approximation properties than standard 'width/depth' networks.
[–]

3/28/26, 9:57 PM Neural Network Architecture Beyond Width and Depth | OpenReview
https://openreview.net/forum?id=36-xl1wdyu&referrer=%5Bthe profile of Shijun Zhang%5D(%2Fprofile%3Fid%3D~Shijun_Zhang1) 3/8
3/28/26, 9:57 PM
Neural Network Architecture Beyond Width and Depth | OpenReview
Strengths And Weaknesses:
The paper is quite rigorous and the definitions are precise and formal. There is some minor repetition in the text
(abstract / introduction), which could be removed to open up some space.
Two weaknesses in my opinion:
1. There is no discussion / intuition / sketch of the proof of Theorem 2.1 in the main text. The construction given in
the Appendix is important as it plays a major role in the theoretical results given in the paper, and a brief
description of it would be valuable to have in the main text (and I believe enough space could be opened up by
removing repeated text, wrapping Fig 3, etc). The organization of the Appendix could also be improved, since a
reader looking for details on the construction used to prove Theorem 2.1 would go to Appendix A, then to B.1
and it is only in B.2 that the construction is given. This is a minor weakness as it is mostly regarding the
organization of the manuscript, and had no impact on my final rating.
2. The discussion on how NestNets relate to models that have been previously proposed and adopted could be
greatly expanded and improved. As the paper states, NestNets can be seen as 'standard' networks but with a
specific parameter sharing scheme (this point could be given more significance in the text, as parameter
sharing seems to be the main ingredient of NestNets and the cause of their expressive power). The scheme is
given by activation layers consisting of element-wise operations on the input, and a repetition of these
activation layers throughout the network.
As for the first property (element-wise maps), there is a connection to convolutions which doesn't seem to be
mentioned: a stack of 1d convolutions with a kernel size of 1 also results in a module that maps an input x = (x1, x2,
..., xk) to u = (f(x1,w), f(x2,w), ..., f(xk,w)) where f will be a deep network that maps R -> R. For image data, we have a
similar relationship with stacks of 2d convolutions with 1x1 kernels and with stacks of depthwise separable 2d
convolutions. Note that stacks of 1x1 or depthwise convolutions are used in some CNN models and, to some extent,
also in transformers (the ReLU MLP blocks following attention layers can be framed as convolutions). To summarize,
this type of parameter sharing has been widely explored in the literature and a more rigorous and complete
discussion on it is required.
For the second property (layer repetition throughout the network), there is also a missing connection to models that
re-use parameters across different layers of the network, most notably models that can be seen as hybrids between
recurrent and non-recurrent networks. Some methods aim to train the recurrence scheme itself, learning how and
when to re-use layer-wise parameters across a deep network. Some examples that should be discussed are:
[1] - Learning Implicitly Recurrent CNNs through Parameter Sharing [2] - ACDC: Weight Sharing in Atom-Coefficient
Decomposed Convolution [3] - Neural Parameter Allocation Search
Lastly, a discussion with Maxout would be valuable, as it can be seen as a learnable activation function but
considerably more expressive than the mentioned ReLU variants.
Questions:
Suggestions are listed in the previous section (Strengths And Weaknesses).
My main question is regarding the practical aspects of NestNets. The construction used to prove Theorem 2.1 seems
to have an effective depth of approximately 
3ns+1
, where by 'effective depth' I mean the diameter of the unrolled
computational graph of the network. For n=10 and s=5, this results in a model that requires 3 million sequential
operations to compute its forward pass (i.e. evaluate the model on a batch of samples), which is on a completely
different scale than the depth of 'standard' networks used in practice and simply impractical. Is it the case that the
approximation power of NestNets indeed rely on networks with an effective depth that outscales the number of
parameters (as n and s increase) and even a single forward pass becomes unfeasible, or is the construction used for
Theorem 2.1 extremely inefficient in terms of effective depth? I understand if a conclusive answer is not possible at
this time but some I'd like to know the authors' thoughts on this.
Limitations:
A discussion on the practical limitations would be valuable (see 'Questions' comments regarding effective depth and
evaluation cost).
Ethics Flag: No
Soundness: 3 good
Presentation: 2 fair
Contribution: 2 fair
Rating: 6: Weak Accept: Technically solid, moderate-to-high impact paper, with no major concerns with respect to
evaluation, resources, reproducibility, ethical considerations.
4/8
https://openreview.net/forum?id=36-xl1wdyu&referrer=%5Bthe profile of Shijun Zhang%5D(%2Fprofile%3Fid%3D~Shijun_Zhang1)
Confidence: 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible,
that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related
work.
Code Of Conduct: Yes
Add Public Comment
02 Aug 2022, 10:23 NeurIPS 2022 Conference Paper9259 Official
Comment Readers:  Everyone Show Revisions (/revisions?
id=fXcMaK7Vc0v)

## Author response — Reviewer zWh2

Response to Reviewer zWh2
NeurIPS 2022 Conference Paper9259 Authors
Comment:
We thank the reviewer for the valuable comments. We will remove some minor repetitions in the text
(abstract/introduction) to open up some space.
We will add a subsection to discuss the idea of proving Theorem 2.1 and the construction of the
corresponding NN in the main text. Therefore, we think it is enough to add pointers/links to Appendix B.2,
which includes the essential construction of the final NN for proving Theorem 2.1.
We completely agree that it is significant to add an in-depth discussion connecting our paper to existing work
from the perspective of parameter sharing. We will add such a discussion based on the suggestions of the
reviewer.
More discussion on the practical aspects will be added.
The architecture of NestNets is flexible. Given a specific problem, we can determine a proper NestNet
architecture, i.e., a proper parameter sharing scheme. In an extreme case, a standard NN is a special case
of a NestNet.
A NestNet (denoted as NN-1) can be expanded to a large standard NN (denoted as NN-2) with many
parameters shared. Clearly, it is numerically cheaper to compute the forward pass of NN-1 than that of a
standard NN with a similar size to NN-2. In the meanwhile, NN-1 has comparable approximation power to
the standard NNs with a similar size to NN-2.
Add Public Comment
[–]

10 Aug 2022, 01:37 NeurIPS 2022 Conference Paper9259 Official
Comment Readers:  Everyone Show Revisions (/revisions?
id=dq_HUt7p72)

## Reviewer follow-up — zWh2

Response to authors
NeurIPS 2022 Conference Paper9259 Reviewer zWh2
Comment:
Thanks for the response.
The revisions described in your response all sound valuable and should improve the paper's
presentation and clarity.
After reading the other reviews, I share some concerns with reviewer wtW7 and believe that adding non
synthetic experiments (even MNIST would be a noticeable improvement) along with in-depth discussion
on the computational cost of NestNets would be very valuable.
Moreover, it seems that my main question regarding the extreme effective depth of the construction in
Theorem 2.1 has not been addressed in the response, and it remains unclear whether the theoretical
advantages of NestNets rely on very unrealistic constructions or not.
Add Public Comment
[–]


## Author response — Reviewer zWh2 (follow-up)

Response to Reviewer zWh2
NeurIPS 2022 Conference Paper9259 Authors
[–]
3/28/26, 9:57 PM Neural Network Architecture Beyond Width and Depth | OpenReview
https://openreview.net/forum?id=36-xl1wdyu&referrer=%5Bthe profile of Shijun Zhang%5D(%2Fprofile%3Fid%3D~Shijun_Zhang1) 5/8
10 Aug 2022, 11:01 (modified: 10 Aug 2022, 11:03) NeurIPS 2022
Conference Paper9259 Official Comment Readers: 
Everyone Show Revisions (/revisions?id=GDYM8mEOLl)
Comment:
Thank you for the further comment. We agree that adding non-synthetic experiments would
improve our paper. We are trying a Fashion-MNIST experiment that compares the performances
of a simple NestNet and a standard NN of almost the same size. The preliminary experimental
results imply that the NestNet outperforms the standard NN and we will continue to improve the
result by adjusting the hyper-parameters.
We will discuss the effective depth of our construction and the theoretical advantages of NestNets
in a new subsection, where we will also provide the proof sketch. In fact, both the effective depth
of our construction and the theoretical advantages of NestNets rely on the idea of parameter
sharing. In our construction of a height- NestNet, we first design a network-generated activation
function  with  parameters and then  is repeatedly used in the final NestNet with
 parameters in total. The repeated use of  leads to much better capacity
of NestNets with  parameters than that of standard NNs with  parameters. The high
NestNet is constructed recursively. We remark that the key point of NestNets is the flexibility of
the parameter sharing scheme. For example, the PReLU activation function just adds a learnable
sharing parameter to ReLU, and thus it is a special and simple case of NestNets. In practice, we
can adopt a proper parameter sharing scheme by choosing a good NestNet architecture based on
the prior knowledge of a specific problem.
Add Public Comment

2
ϱ O(n) ϱ
O(n)+O(n)=O(n) ϱ
O(n) O(n)
07 Jul 2022, 15:28 (modified: 04 Aug 2022, 15:02) NeurIPS 2022 Conference
Paper9259 Official Review Readers:  Everyone Show Revisions (/revisions?
id=pBjei96nLSJ)

## Official review — Reviewer wtW7

Official Review of Paper9259 by Reviewer wtW7
NeurIPS 2022 Conference Paper9259 Reviewer wtW7
Summary:
The paper presents a novel three-dimensional NN architecture. An additional dimension called height is introduced
to empower the capacity of neural networks. A simple 257 experiment demonstrates the numerical advantages of
the proposed method. The proof seems to be solid. However, I am not an expert in theory so I cannot judge the
contribution of the proof. Overall, I think the idea of the paper is novel.
Strengths And Weaknesses:
Strengths
The three-dimensional NN architecture is a novel concept. The authors present theoretical proof to show their
advantage with ReLu activations. The proof seems to be solid.
Weaknesses
Although the method does not add extra parameters, the proposed three-dimensional NN significantly
introduces an extra computational burden to iteratively activate the neurons (e.g., FLOPs during inference). It is
good to present the computational burden of training and inference, and compare the performance against the
computational burden.
The results of commonly-used benchmarks are preferred such as mnist, cifar, and ImageNet.
Questions:
See Weakness
Limitations:
There is no computational cost analysis and comparison, or experiments on real-world datasets.
Ethics Flag: No
Soundness: 2 fair
Presentation: 4 excellent
[–]

3/28/26, 9:57 PM Neural Network Architecture Beyond Width and Depth | OpenReview
https://openreview.net/forum?id=36-xl1wdyu&referrer=%5Bthe profile of Shijun Zhang%5D(%2Fprofile%3Fid%3D~Shijun_Zhang1) 6/8
Contribution: 2 fair
Rating: 5: Borderline accept: Technically solid paper where reasons to accept outweigh reasons to reject, e.g.,
limited evaluation. Please use sparingly.
Confidence: 2: You are willing to defend your assessment, but it is quite likely that you did not understand the
central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were
not carefully checked.
Code Of Conduct: Yes
Add Public Comment
02 Aug 2022, 10:24 NeurIPS 2022 Conference Paper9259 Official
Comment Readers:  Everyone Show Revisions (/revisions?
id=uAGSNKhC41N)



## Author response — Reviewer wtW7

Response to Reviewer wtW7
NeurIPS 2022 Conference Paper9259 Authors
Comment:
We thank the reviewer for pointing out the contributions of our paper. The goal of our paper is to design a
new NN architecture by introducing an additional dimension height to achieve a better approximation error
than standard NNs. The focus of this paper is on the theoretical proof, and hence we only use a simple
experiment as a proof of concept for our new NNs. We believe our new NNs can be further developed and
applied to real-world applications. However, it is challenging to estimate the computational burden and tune
hyper-parameters for applying our new NNs to commonly-used benchmark datasets, and hence they are left
for future work.
Add Public Comment
[–]

04 Aug 2022, 15:01 (modified: 04 Aug 2022, 15:04) NeurIPS 2022
Conference Paper9259 Official Comment Readers:  Everyone Show
Revisions (/revisions?id=WejLyRBeHPU)
Need Computational Cost Comparison for the simple task such
as mnist
NeurIPS 2022 Conference Paper9259 Reviewer wtW7
Comment:
I still have concerns about the computational cost. In fact, It is very easy to compute the FLOPs and
measure the latency on the smallest commonly-used dataset mnist (CPU experiments are just enough).
It is ok that your main contribution is theory. However, I do not figure out the benefits of the theory or
further promising directions revealed by the theory. Thus, I have concerns that the limited capacity
improvement is worth sacrificing the huge computational resource. I decided to decrease my rating but
keep the positive score.
Add Public Comment
[–]

07 Aug 2022, 00:35 NeurIPS 2022 Conference Paper9259 Official
Comment Readers:  Everyone Show Revisions (/revisions?
id=ICW5bhsUEra)


## Author response — Reviewer wtW7 (follow-up)

Response to Reviewer wtW7 
NeurIPS 2022 Conference Paper9259 Authors
Comment:
We agree that the computational cost of NestNets is a concern.
Let us first compare the training time. For the experimental example in the paper, it takes about
470 and 780 seconds to train the standard and NestNet models, respectively. The numerical
accuracy relies on approximation and generalization errors. The numerical example in the paper
allows us to generate sufficiently many samples to trivialize the generalization error so that we
can make sure that the improvement of accuracy is due to the reduction of the approximation
error as shown by our theory. This example also implies the computational cost of NestNets is
[–]

3/28/26, 9:57 PM Neural Network Architecture Beyond Width and Depth | OpenReview
https://openreview.net/forum?id=36-xl1wdyu&referrer=%5Bthe profile of Shijun Zhang%5D(%2Fprofile%3Fid%3D~Shijun_Zhang1) 7/8
within control, at least for the simple ones. We agree that it would be better to use standard
benchmark datasets, e.g., Fashion-MNIST. It is challenging and requires significant work to
develop numerical techniques in tuning the hyper-parameters to show the advantage of NestNet
models numerically with a limited sample size. Nevertheless, we have also tried to use the
Fashion-MNIST dataset to compare the performances of a standard ReLU NN and a simple
NestNet that only some of its neurons have nested activation functions and other neurons are
activated by ReLU. The time of training the NestNet model is only a little ( %) more than that
of training the standard model. We also agree that it is good to estimate the computational cost
for a NestNet architecture as an example. However, it is challenging to theoretically estimate the
exact computational burden for a general NestNets since the architecture of NestNets is flexible
and can be pretty complicated sometimes. The activation function of each neuron of a NestNet
can be as simple as ReLU or as complicated as a high sub-NestNet. It is of interest to explore a
(recursive) formula to describe the computational cost of NestNets.
Finally, we would like to point out again that the main goal of this paper is to introduce a concept
of height to NN architectures beyond depth and width as the first step of this new research
direction. Our theory implies that the height of NNs would improve the approximation power. This
gives us some hope and opens many possibilities and interesting questions to be further
explored, which are left for future work.
Add Public Comment
≤10

## OpenReview site footer (Paper 1)

About OpenReview (/about)
Hosting a Venue (/group?
id=OpenReview.net/Support)
All Venues (/venues)
Sponsors (/sponsors)
News (/group?
id=OpenReview.net/News&referrer=[Homepage]
(/))
FAQ (https://docs.openreview.net/getting
started/frequently-asked-questions)
Contact (/contact)
Donate (/donate)
Terms of Use (/legal/terms)
Privacy Policy (/legal/privacy)
OpenReview (/about)is a long-term project to advance science through improved peer review with legal nonprofit status.
We gratefully acknowledge the support of theOpenReview Sponsors (/sponsors). © 2026 OpenReview ;;;

---

# On Enhancing Expressive Power via Compositions of Single Fixed Size ReLU Network

## Paper 2 — ICML 2023 (Zhang, Lu, Zhao)

**OpenReview:** [forum?id=uIOw2ZE1U8](https://openreview.net/forum?id=uIOw2ZE1U8) · **PDF id:** `uIOw2ZE1U8` · **Venue:** ICML 2023 (poster)

**Authors:** Shijun Zhang, Jianfeng Lu, Hongkai Zhao

**Listing dates:** published 25 Apr 2023; last modified 15 Jun 2023.

**Keywords:** function composition, parameter sharing, deep neural network, dynamical systems, function approximation.

### Abstract (LaTeX — cleaned from the OpenReview export)

The original OpenReview copy-paste had broken line breaks and misplaced symbols (`L2`, `f/ixed`, split composition of $g$, etc.). The following block is the same mathematical content in proper LaTeX (verify wording against the official PDF if you need a camera-ready match).

```latex
% On Enhancing Expressive Power via Compositions of Single Fixed-Size ReLU Network
% Abstract (notation cleaned; check against the official PDF for final wording)

This paper explores the expressive power of deep neural networks through the framework of
function compositions. We demonstrate that repeated compositions of a single fixed-size ReLU
network exhibit surprising expressive power, despite the limited expressive capabilities of the
individual network itself. Specifically, we prove by construction that
\[
  L_2 \circ g^{\circ r} \circ L_1
\]
can approximate $1$-Lipschitz continuous functions on $[0,1]^d$ with approximation error
$O(r^{-1/d})$, where $g$ is realized by a fixed-size ReLU network, $g^{\circ r}$ denotes the
$r$-fold composition of $g$, and $L_1,L_2$ are affine linear maps matching the dimensions.
Furthermore, we extend this result to generic continuous functions on $[0,1]^d$, with error
quantified via the modulus of continuity. Our results show that a continuous-depth network
generated by a dynamical system can have strong approximation power even if the dynamics map
is time-independent and realized by a fixed-size ReLU network.
```

Financial Aid:

shijun.math@outlook.com
Paper Checklist Guidelines:

I certify that all co-authors of this work have read and commit to adhering to the Paper
Checklist Guidelines, Call for Papers and Publication Ethics.
Verify Author Names:

in the camera-ready PDF.
No Additional Revisions:
My co-authors have confirmed that their names are spelled correctly both on OpenReview and

I understand that this submission is the final camera ready paper and that there will not be
another opportunity to revise it. I have verified with all authors that they approve of this version.
Pdf Appendices:

My camera-ready PDF file contains both the main text (not exceeding the page limits) and all
appendices that I wish to include. I understand that any other supplementary material (e.g., separate files uploaded to
OpenReview) will not be visible in the PMLR proceedings.
Latest Style File:

I have compiled the camera ready paper with the latest ICML2023 style files
(https://media.icml.cc/Conferences/ICML2023/Styles/icml2023.zip
(https://media.icml.cc/Conferences/ICML2023/Styles/icml2023.zip)) and checked that the compiled PDF shows the page
number at the bottom of each page.
Paper Verification Code:
Permissions Form:


NGEyY
pdf (/attachment?id=uIOw2ZE1U8&name=permissions_form)
Submission Number: 6689
Filter by reply type...
Filter by author...
Sort: Newest First
 Ev
eryone
Program Chairs
Submission6689 Area...
Search keywords...
Submission6689 Authors
Submission6689...
−
＝
≡

10 / 10 replies shown
Submission6689...
Submission6689...
Submission6689...
1/7
https://openreview.net/forum?id=uIOw2ZE1U8&referrer=%5Bthe profile of Shijun Zhang%5D(%2Fprofile%3Fid%3D~Shijun_Zhang1)
Submission6689...


## Decision — ICML 2023

Paper Decision
Decision byProgram Chairs 25 Apr 2023, 04:09 (modified: 25 Apr 2023, 05:39)
Program Chairs, Authors Revisions (/revisions?id=10WGg2gJ6O)

 
Decision: Accept (Poster)
Comment:
This paper makes significant contributions to understanding the approximation power of neural networks. More
specifically, authors show that a ReLU network of  depth can approximate a Lipschitz function of input
dimension  with  rate. They further extend these results to generic continuous functions on  and
quantify the approximation error in this case, with the modulus of continuity.
One drawback of the result is the dimension dependence. The approximation error bound deteriorates
significantly for large , therefore the method still suffers from the curse of dimensionality. However, the
dependence on the depth is still a good contribution.
Three reviewers all have positive opinion on this paper, which the AC agrees with. Therefore, I recommend
including this paper to the conference program.
O(r)
d O(r−1/d) [0,1]d
d

## Official review — Reviewer R9Yq

Official Review of Submission6689 by Reviewer R9Yq
Official ReviewbyReviewer R9Yq 06 Mar 2023, 08:23 (modified: 14 Mar 2023, 21:36)
Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Reviewer R9Yq, Authors
Revisions (/revisions?id=4MH4DlODIZ)



Summary:
This paper studies the expressive power of deep neural networks from the perspective of function compositions.
We show that repeated compositions of a single fixed-size ReLU network can produce super expressive power.
And the results in this paper reveal that a continuous-depth network generated via a dynamical system has good
approximation power even if its dynamics function is time-independent and realized by a fixed-size ReLU network.
Strengths And Weaknesses:
The paper is well-written and well-organized. The proposed construction scheme is novel and well-motived.
Questions:
Can the result in this paper be generalized to the case where the domain is unbounded ?
Limitations:
It is better to verify the result with toy numercial examples.
Ethics Flag: No
Soundness: 4 excellent
Presentation: 4 excellent
Contribution: 4 excellent
Rating: 8: Strong Accept: Technically strong paper, with novel ideas, excellent impact on at least one area, or
high-to-excellent impact on multiple areas, with excellent evaluation, resources, and reproducibility, and no
unaddressed ethical considerations.
Confidence: 5: You are absolutely certain about your assessment. You are very familiar with the related work and
checked the math/other details carefully.
Code Of Conduct: Yes

## Author rebuttal — Reviewer R9Yq

Rebuttal by Authors

Rebuttal
byAuthors ( Shijun Zhang (/profile?id=~Shijun_Zhang1), Jianfeng Lu (/profile?id=~Jianfeng_Lu1),
zhao@math.duke.edu (/profile?id=zhao@math.duke.edu))
19 Mar 2023, 09:01 (modified: 20 Mar 2023, 22:34)
Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors
Revisions (/revisions?id=I8mqA0xnSg)
−
＝




≡
3/28/26, 9:56 PM On Enhancing Expressive Power via Compositions of Single Fixed-Size ReLU Network | OpenReview
https://openreview.net/forum?id=uIOw2ZE1U8&referrer=%5Bthe profile of Shijun Zhang%5D(%2Fprofile%3Fid%3D~Shijun_Zhang1) 2/7
Rebuttal:
We would like to express our gratitude to the reviewer for providing a positive evaluation of our paper.
Our result cannot be generalized to the general case with an unbounded domain in terms of the
supremum norm. It appears to be impossible to use ReLU networks to approximate generic continuous
functions uniformly well on . For instance, one could demonstrate that
, where  denotes the hypothesis space generated by ReLU
networks, which is actually the function space of all continuous piecewise linear functions on .
To provide a proof of concept, we have included two simple numerical examples. We agree with the
reviewer that including additional numerical examples, such as those related to (Fashion-)MNIST
classification, would be beneficial. However, it is important to note that our result is not readily
generalizable to convolutional neural networks (CNNs) that are commonly employed in such
applications. Extending our result to other network architectures, including CNNs, would require
significant effort and is therefore reserved for future research.
Rd
infϕ∈Hsupx∈R|ϕ(x)−sinx|≥1 H
R

## Official review — Reviewer SpYd

Official Review of Submission6689 by Reviewer SpYd
Official ReviewbyReviewer SpYd 05 Mar 2023, 12:20 (modified: 22 Mar 2023, 06:23)
Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Reviewer SpYd, Authors
Revisions (/revisions?id=PYzWz9k1eb)



Summary:
This work shows that a single fixed-size ReLU network of width  composed  times can produce
approximation power that scales as . They also show a connection with dynamical systems and verify
their results with some numerical experiments. The key aspect to note here is that the number of parameters
remains fixed with increasing depth and still leads to increasing approximation power - which is quite surprising.
Strengths And Weaknesses:
Strengths:
1. The paper is well written with good proof sketches and well explained intuition.
2. The authors discuss the rich history of universal approximation well and do a thorough treatment of the
related work.
3. The result is quite impressive and quite surprising at first glance. It also explains the remarkable success of
parameter sharing schemes used in practice.
Weaknesses:
1. While the bound looks quite impressive to begin with, it does not escape the curse of dimensionality.
Theorem 1.1 requires  compositions of a -wide network to reach -accuracy. Despite having a constant
number of trainable parameters, just computing this function is quite expensive.
2. The dependence on the modulus of continuity seems somewhat unnecessary - I am not able to get a good
intuition of whether this would lead to a reasonable bound for functions that are not Lipschitz. If the authors
have some canonical examples, that would be useful.
3. The numerical experiments section are somewhat lacking in that the improved performance with increasing 
is not surprising given the result. However, I understand that this is primarily a theoretical contribution and
therefore this is a minor criticism.
Post rebuttal: I am satisfied with the authors' response and I am increasing my score.
Questions:
1. I was thinking about some kind of intuition how composing a fixed ReLU network can improve the
approximation power and I think one can somewhat understand this using polynomials. Considering a
polynomial . Considering  would increase the degree of the polynomial and
therefore improve its approximation power. Does this intuition make sense? I understand that this is a gross
simplification of your result which requires fairly heavy technical tools to prove, but I believe the non-linearity
of ReLU is what allows this result.
2. The authors use the term "super-approximation power" repeatedly in the paper. Is this a technical term that
is commonly used in the literature? If not, please define it.
3. Prop 3.3 has a typo: *there -> three.
O(d) r
O(r−1/d
)
ϵ−d d ϵ
r
fa,b(x)=ax2+bx f∘r
3/28/26, 9:56 PM On Enhancing Expressive Power via Compositions of Single Fixed-Size ReLU Network | OpenReview
https://openreview.net/forum?id=uIOw2ZE1U8&referrer=%5Bthe profile of Shijun Zhang%5D(%2Fprofile%3Fid%3D~Shijun_Zhang1) 3/7
4. The figures in Section 4 are missing y labels. Please add them to improve readability. Also, I would suggest
adding a more detailed caption and using dashed-lines or some other mechanism to make the plots more
distinguishable.
5. Can you add some comparisons with wider/deeper networks in the numerical experiments to show
something of a trade-off between number of trainable parameters and total parameters?
6. I would recommend re-ordering the proofs in the appendix to be "bottom-up". Since every theorem assumes
results that follow it, it makes reading the proofs quite cumbersome.
Limitations:
I would recommend the authors add a small note describing the exponential dependence of  on the dimension
 to reach  error. This is not too big a deal according to me since as the authors remark, the number of trainable
parameters is still just .
Ethics Flag: No
Soundness: 4 excellent
Presentation: 3 good
Contribution: 3 good
Rating: 8: Strong Accept: Technically strong paper, with novel ideas, excellent impact on at least one area, or
high-to-excellent impact on multiple areas, with excellent evaluation, resources, and reproducibility, and no
unaddressed ethical considerations.
Confidence: 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible,
that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related
work.
Code Of Conduct: Yes
r
d ϵ
O(d)

## Author rebuttal — Reviewer SpYd (W1–W3)

Rebuttal by Authors

Rebuttal
byAuthors ( Shijun Zhang (/profile?id=~Shijun_Zhang1), Jianfeng Lu (/profile?id=~Jianfeng_Lu1),
zhao@math.duke.edu (/profile?id=zhao@math.duke.edu))
19 Mar 2023, 10:23 (modified: 20 Mar 2023, 22:34)
Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors
Revisions (/revisions?id=xcjE21jmk7)
−
＝




Rebuttal:
We appreciate the reviewer for providing valuable and insightful comments. Please find below our
responses to each of the Weaknesses (W), Questions (Q), and Limitations (L).
W1:   We acknowledge that our result still suffers from the curse of dimensionality, as 
compositions are required to achieve -error. Approximating all -Lipschitz continuous functions
well on  is an intrinsically high-dimensional problem. To the best of our knowledge, it is
impossible to fundamentally overcome (not just transfer) the curse of dimensionality for such a
problem unless one considers a much smaller target function space like the Barron space. We also
agree that it is expensive to compute  (or its sub-gradient) if  is large, where  is generated by a
fixed-size ReLU network. It is evident that  has good approximation power for large ,
where  are realized by fixed-size ReLU networks. However, it is highly non-trivial to show
that  also has good approximation power, which is a key contribution of our paper. Moreover,
compared to  ( parameters), computing  ( parameters) uses significantly
less memory and is much more computationally efficient.
W2:   We recognize the need to extend our result to generic continuous functions on  with
the error characterized by the modulus of continuity. For instance, we can define
 via  and  for . Then, we have
 for any , even though  is not Lipschitz (Hölder) continuous.
W3:   The primary goal of our paper is to demonstrate that repeated compositions of a single fixed
size ReLU network could enhance the approximation power. As a proof of concept, our experiments
aim to numerically verify our theoretical result. As shown in Section 4, the experiment results do
improve as  increases. We agree that the improvement is not surprising given that deep learning
ε−d
ε 1
[0,1]d
g∘r r g
g1∘⋯∘gr r
g1,⋯,gr
g∘r
g1∘⋯∘grO(r) g∘rO(1)
[0,1]d
f:[0,1]→[0,1/2] f(0)=0 f(x)=1/(2−lnx) x∈(0,1]
ωf(r)≤1/(2−lnr) r>0 f
r
≡
3/28/26, 9:56 PM On Enhancing Expressive Power via Compositions of Single Fixed-Size ReLU Network | OpenReview
https://openreview.net/forum?id=uIOw2ZE1U8&referrer=%5Bthe profile of Shijun Zhang%5D(%2Fprofile%3Fid%3D~Shijun_Zhang1) 4/7
3/28/26, 9:56 PM
On Enhancing Expressive Power via Compositions of Single Fixed-Size ReLU Network | OpenReview
optimization is notoriously challenging, with issues such as local minima, saddle points, and
vanishing gradients. The key difficulty in our experiments is identifying the global minimizer,
particularly for large .
r
−
＝
≡

## Author rebuttal — Reviewer SpYd (Q1–Q6, L)

Rebuttal by Authors





Rebuttal
by Authors ( Shijun Zhang (/profile?id=~Shijun_Zhang1), Jianfeng Lu (/profile?id=~Jianfeng_Lu1),
zhao@math.duke.edu (/profile?id=zhao@math.duke.edu))
19 Mar 2023, 10:24 (modified: 20 Mar 2023, 22:34)
Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors
Revisions (/revisions?id=5oPn2nG9LZ)
Rebuttal:
Q1:   We think the reviewer's intuition makes sense. It is of interest to explore the architecture of
repeated compositions of a polynomial, the idea of which is used in constructing the Mandelbrot
set (https://en.wikipedia.org/wiki/Mandelbrot_set). The goal of our paper is to prove that repeated
compositions of a fixed-size ReLU network can enhance approximation power. Additionally, we
would like to emphasize the critical roles played by the non-linearity of ReLU and various
techniques (e.g., bit extraction) in our proof.
y
Q2:   To the best of our knowledge, the term "super approximation power'' does not have a specific
meaning in the literature. We just use the word "super" to describe good approximation power.
Q3:   We have corrected the typo in Proposition 3.3 by changing "there" to "three."
Q4:   We will revise the figures in Section 4 as per the reviewer's suggestions. Specifically, we will
add  labels, more detailed captions, and different line styles to the figures in Section 4 to make
them more distinguishable.
Q5:   We are currently conducting experiments using wider/deeper networks, and we will
incorporate the results of these experiments into the revised version.
Q6:   We will reorganize the proofs in the revised version based on the reviewer's suggestion.
L:   We will add a note to describe the dependence of  on  for reaching -error and to discuss the
connection between our results and existing ones in terms of the curse of dimensionality.
−
＝ 

≡

## Thread — after authors' rebuttal

Replying to Rebuttal by Authors

## Reviewer follow-up — SpYd

Response to authors
Official Comment byReviewer SpYd


r

d
ε
22 Mar 2023, 06:10 (modified: 22 Mar 2023, 06:22)
Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors
Revisions (/revisions?id=pkHgKXVmj8)
Comment:
I thank the authors for their detailed response.
W1: I agree that approximating all 1-Lipschitz functions on  
[0, 1]d
seems to be hard. As the authors
suggest, the results I am familiar with all consider restricted spaces. I would suggest the authors
add a note regarding the curse of dimensionality and comparison to approximation results that
beat it (on restricted spaces) - which it looks like you have already mentioned under L. Thanks!
W2: Thank you for the simple example. That helps.
Q2: If that is the case, I recommend the authors to either rephrase it so that it is more precise.
Maybe something like "approximation power that improves with  even though the number of
parameters remains the same" or create define it precisely in the beginning of the paper.
Otherwise, it is hard to understand what it means while reading the paper.
r
I am satisfied with the authors' responses and will therefore increase my score to 8.
−
＝ 

≡

Replying to Response to authors
Response to Reviewer SpYd
5/7
https://openreview.net/forum?id=uIOw2ZE1U8&referrer=%5Bthe profile of Shijun Zhang%5D(%2Fprofile%3Fid%3D~Shijun_Zhang1)
3/28/26, 9:56 PM
On Enhancing Expressive Power via Compositions of Single Fixed-Size ReLU Network | OpenReview





Official Comment
by Authors ( Shijun Zhang (/profile?id=~Shijun_Zhang1), Jianfeng Lu (/profile?id=~Jianfeng_Lu1),
zhao@math.duke.edu (/profile?id=zhao@math.duke.edu))
22 Mar 2023, 08:53 (modified: 22 Mar 2023, 08:53)
Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors
Revisions (/revisions?id=Q7Bgdc1ip0)
Comment:
We thank the reviewer for the further comment. We will include a note in the revised version to discuss
the connection between our results and existing ones (on restricted target function spaces), especially
with regards to the curse of dimensionality. We fully acknowledge that our current statement on "super
approximation power" is not clear enough within its context. Therefore, we will strive to rephrase it in a
more comprehensible and pertinent manner. Once again, we appreciate the reviewer's valuable input
and are committed to enhancing the quality of our paper based on all the suggestions.

## Official review — Reviewer zdJr

Official Review of Submission6689 by Reviewer zdJr
Official Review by Reviewer zdJr



01 Mar 2023, 01:34 (modified: 14 Mar 2023, 21:36)
Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Reviewer zdJr, Authors
Revisions (/revisions?id=kJVFoTayF4)
Summary:
The authors study the universal approximation problem of neural networks. The main contribution of this paper
is to show that, repeated compositions of a single fixed-size network have strong expressive power, for example,
they can approximate 1-Lipschitz continuous functions on  
[0, 1]d
with an error  
O(r−1/d)
if there are 
compositions of a single fixed-size network. Furthermore, they build the connection between their results and
dynamical systems.
Strengths And Weaknesses:
Originality: The related works are adequately cited. The main results in this paper will certainly help us have a
better understating of the universal approximation property of deep neural networks from a theoretical way. I
have checked the technique parts and found that the proofs are solid. The main result, which derives the error
term for repeated-composition networks to approximate given functions in 
, is a non-trivial extension of
previous results in this field. In summary, I think this paper is a good contribution to the machine learning
community.
Quality: This paper is technically sound.
Clarity: This paper is clearly written and well organized. I find it easy to follow.
Significance: I think the results in this paper are significant, as explained above.
Questions:
Limitations:
C([0,1]d)
r
It would be more interesting if the authors could extend their results to more activation functions and more
architectures used in practice.
Yes, the authors have adequately addressed the limitations and potential negative societal impact of their work.
Ethics Flag: No
Soundness: 4 excellent
Presentation: 3 good
Contribution: 3 good
Rating: 6: Weak Accept: Technically solid, moderate-to-high impact paper, with no major concerns with respect to
evaluation, resources, reproducibility, ethical considerations.
Confidence: 3: You are fairly confident in your assessment. It is possible that you did not understand some parts
of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not
carefully checked.
Code Of Conduct: Yes
−
＝

## Author rebuttal — Reviewer zdJr

Rebuttal by Authors
6/7
https://openreview.net/forum?id=uIOw2ZE1U8&referrer=%5Bthe profile of Shijun Zhang%5D(%2Fprofile%3Fid%3D~Shijun_Zhang1)
3/28/26, 9:56 PM

On Enhancing Expressive Power via Compositions of Single Fixed-Size ReLU Network | OpenReview
≡




Rebuttal
by Authors ( Shijun Zhang (/profile?id=~Shijun_Zhang1), Jianfeng Lu (/profile?id=~Jianfeng_Lu1),
zhao@math.duke.edu (/profile?id=zhao@math.duke.edu))
19 Mar 2023, 09:23 (modified: 20 Mar 2023, 22:34)
Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors
Revisions (/revisions?id=4e2LfSRsRK)
Rebuttal:
We would like to express our sincere appreciation to the reviewer for acknowledging the contributions
of our paper. We completely agree with the reviewer’s suggestion to broaden the scope of our paper by
generalizing our results to other activation functions (e.g., the tanh function) or other network
architectures (e.g., convolutional neural networks). However, we have attempted to generalize our
f
indings to these cases and have found that they require significant effort. Therefore, we choose to
leave them as future research directions. We hope that the reviewer could appreciate the challenges
involved in such generalizations and the need for further investigation in these areas.

## OpenReview site footer (Paper 2)

About OpenReview (/about)
Hosting a Venue (/group?
id=OpenReview.net/Support)
All Venues (/venues)
Sponsors (/sponsors)
News (/group?
id=OpenReview.net/News&referrer=[Homepage]
(/))
OpenReview
FAQ (https://docs.openreview.net/getting
started/frequently-asked-questions)
Contact (/contact)
Donate (/donate)
Terms of Use (/legal/terms)
Privacy Policy (/legal/privacy)
(/about) is a long-term project to advance science through improved peer review with legal nonprofit status.
We gratefully acknowledge the support of the
OpenReview Sponsors
(/sp
