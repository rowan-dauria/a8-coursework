# MY WRITING STYLE
*Use these answers to application questions to align your writing style with mine. The specific content from this file is NOT to be referenced.*

## General Guidelines
- Do not us em-dashes
- Do not use "it's not <thing>, it's <other thing>" phrases.
- Write in british english

**Why do you want to work on reducing risks from advanced AI through this fellowship? (1200 characters)\***

I am interested in the Alignment Fellowship because AI Safety buy-in from influential actors is dangerously low. The Fellowship will empower me to change this in 3 ways.

Firstly, ERA’s mentorship model is the ideal framework to develop my master’s dissertation on mechanistic interpretability. For me, the shortest path to impact is to build on the findings of my dissertation, towards a published paper on knowledge elicitation. The guidance of ERA’s expert mentors will be invaluable for this project. Secondly, the Fellowship will build my network within the AI Safety community, enhancing my capacity to inform AI policy makers and build support for the movement with the wider public. Finally, ERA’s role at the intersection of technical safety and governance is well-placed for impactful connections with like minded peers in the policy space, providing the networks and introductions to translate technical insights into actionable policy as quickly as possible. In addition, ERA’s position in the Cambridge community will enable me to raise awareness with future leaders who are optimally positioned to drive change.

**What (if any) relevant experience do you have for working in the track you have selected? (800 characters)\***

*This could include projects you've worked on, reading you've done, and your experience with software/programming (e.g., experience using Inspect). Please link us to an example of your past work or code, if appropriate. For technical AI safety or technical AI governance fellows, if you have experience with specific software or libraries relevant to your interests, you can mention this here.*

*As we also have your CV, please focus on highlighting which parts of your experience are particularly relevant and why, as opposed to just re-stating your experience in this question.*

My dissertation on circuit tracing with sparse autoencoders is the basis of my expertise for the technical track. Through the project, I have trained SAEs, built attribution graphs and performed activation steering on Qwen3 models. I have also become comfortable with training activation oracles to compare with SAE methods, specifically on LLM fine-tuning for control on political discourse.

Alongside this, I'm currently completing the Cambridge AI Safety Hub Alignment Fellowship, which has deepened my understanding of technical and governance dimensions of AI safety through workshops and hands-on assignments.

At Lindus Health, I built and integrated an LLM-powered documentation system adopted across the deployment team, giving me applied experience working with language models in production.

**\[Technical AI Safety Question\] Methodological Improvements (800 characters)\***

*Read the following report by Model Evaluation & Threat Research (METR): [Task-Completion Time Horizons of Frontier AI Models](https://metr.org/time-horizons/).*

*In the report, METR measures synthetic tasks at 50% success rate. Why did they choose this method to measure model capabilities? What are the limitations of this method? What are the differences between their environments and real-world tasks?*

METR chose this method because it aligns with their mission to convey AI risks to institutional audiences. It has three key features: it quantifies AI progress over years; it benchmarks against human completion time, giving non-technical decision-makers an idea of AI progress; and its modular task suite allows new tests to be added for the latest models.

However, building tasks that stretch frontier capabilities becomes harder as horizons extend, impacting accuracy of estimating THs. Large error bars also weaken confidence in estimates, Opus 4.6 has 95% CIs of 5–66 hours.

A key difference from real-world tasks is the narrow domain coverage: SWE, AI R\&D, and cybersec. Their tasks are also self-contained, unlike real work which requires collaboration, requirements gathering, and subjective judgment.

**What is one potential project or research question you would be interested to work on as part of ERA? (2000 characters)\***

*You don’t need a fully scoped research idea at this stage, and nothing you write here is a commitment. If you're accepted to the programme, your project can evolve substantially (and may differ entirely) after you meet your research manager and mentors. However, it is a useful exercise to brainstorm what you could actually work on, and it helps us understand your research interests.*

*Please make an effort to highlight key sub-questions and the theory of change (how would the research ultimately lead to reducing large-scale catastrophic risks from advanced AI?).*

There is a small body of research on eliciting latent knowledge (ELK) from LLMs. Current work in the field shows promising results using SAE methods for ELK, but the current methodology ([Cywiński et al., 2025](https://arxiv.org/abs/2505.14352)) relies on a simple “taboo” model that is narrowly fine-tuned to avoid a specific word. Qwen, and other Chinese models, are extensively trained with SFT and RLHF to lie, deflect and shut down conversations on topics disfavoured by the CCP. These models provide a rich test bed for ELK research in more realistic and robust contexts.

So far, there is nothing in the literature testing mechanistic approaches to understand the biases of Chinese LLMs. A recent [MATS](https://www.lesswrong.com/posts/7gp76q4rWLFi6sFqm/test-your-interpretability-techniques-by-de-censoring-1) project provides a range of scenarios where Chinese LLMs can be tested for refusal and CCP aligned responses. A key limitation of the project is the assumption that models know the correct information and choose to lie, rather than faithfully recounting biased training data. However, this does not establish whether the model internally represents such information as credible knowledge it actively suppresses, or as "foreign propaganda" worthy of criticism. This distinction between a brainwashed and deceptive model is critical for alignment.

I propose using circuit-level interpretability to investigate whether Chinese LLMs maintain truthful internal representations upstream of their refusal behaviour. My preliminary analysis of a model discussing Uighur persecution revealed SAE features encoding factual knowledge of documented abuses, linked to features activating on government censorship patterns, before the model's output denied any persecution.

This matters for catastrophic risk because if models can be trained to systematically conceal knowledge they possess, and we lack reliable methods to detect this, oversight mechanisms fundamental to safe deployment break down. Developing robust ELK techniques on real-world deceptive models directly strengthens our ability to detect misalignment.

In this open-ended section we're going to ask you to write a **response** to **_one_** of the following pieces. Some of these are quite long (and none are particularly short). You should choose the one you're most excited to read and then mull over. Because this is not a quiz, you need not read the entire piece -- you can be strategic in which sections you read. In the "Response" box below, you should **write between 300 and 600 words** on the reading. You can imagine that the incentive structure and what constitutes a good response is similar to that of an advanced undergraduate or graduate-level seminar. We're deliberately leaving the task underspecified to give potential mentees more degrees of freedom.

**Stimuli choices** (choose one):
[A Pragmatic Vision for Interpretability](https://www.lesswrong.com/posts/StENzDcD3kpfGJssR/a-pragmatic-vision-for-interpretability) <--- I chose this one

### Response
*Write this to a reader who is familiar with the text. That is, do **not** write an explainer or summary*

The post is a well-motivated call-to-arms for the interpretability community to change direction in light of shortening timelines. But given the major downgrade in prospects for white-box interpretability, and the post's framing of comparative advantage for experienced interpretability researchers, it raises the question: what to do if you have little or no comparative advantage?

The combined effect of shortening timelines for AGI and disappointing results from ambitious interpretability research motivates a paradigm shift in how interpretability work is conducted. Transitioning to an approach with tighter feedback loops, easily measurable outcomes and more tangible objectives is the best way to approach interpretability research in the face of impending AGI. The piece has changed my perception of interpretability and will inform how I develop my research career, specifically how I develop an interpretability toolkit alongside other techniques.

However, the piece has a significant omission. Despite downgrading the ambitions of interpretability as a research domain and recommending a change of direction, it doesn't re-evaluate the impact of white-box interpretability relative to other domains of technical AI Safety. Their motivating case on Jack Lindsey's eval awareness neatly illustrates this. Lindsey uses linear probes and contrastive pairs to great effect, but these are far from cutting-edge white-box methods. By changing interpretability from a standalone research agenda to "one more tool in the toolbox", they cast doubt on whether new researchers should develop skills in  interpretability, rather than entirely different domains like safe RL, AI control, or others. DeepMind's pivot from exclusively working on mechanistic interpretability to "mechanistic OR interpretability" is a striking indictment of the field. Given the team's motivating case and examples of future work, inexperienced researchers with short AGI timelines should treat interpretability as a set of tools to be deployed as needed, rather than focussing on building their research expertise solely around interpretability.

I agree with the over-arching thesis of the post, and think Richard Ngo's criticism of focussing on "category 1 problems" is naive. With current AI progress, focussing on ambitious and risky safety research is analogous to "preventing all future wars" in the lead-up to a nuclear war - what needs to be addressed is the impending nuclear war. In this framework for AI risk, pragmatic interpretability is the best option for avoiding catastrophe. However, he makes one interesting point, there is a risk that new researchers don't realise that ambitious interpretability is even an option. Whilst I think almost all researchers should pivot to pragmatic interpretability, as models become increasingly capable, ambitious interpretability may become possible with AI assistance, and it would be a disaster if this opportunity arose with no one to pursue it.

Although the post recalibrated my trajectory towards interpretability as one component of a larger toolkit, the lack of explicit advice where there is no comparative advantage gives me some uncertainty for my big-picture strategy. Also, I think there is a secondary risk of developing tunnel-vision, preventing us from tackling ambitious interpretability when advanced AI affords us the opportunity.