/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
package com.example.distilbert

import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.speech.tts.TextToSpeech
import android.text.Editable
import android.text.Spannable
import android.text.SpannableString
import android.text.TextWatcher
import android.text.method.ScrollingMovementMethod
import android.text.style.BackgroundColorSpan
import android.util.Log
import android.view.KeyEvent
import android.view.View
import android.view.View.OnFocusChangeListener
import android.view.inputmethod.InputMethodManager
import android.widget.ProgressBar
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.Observer
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.example.distilbert.gemma.ChatViewModel
import com.example.distilbert.gemma.InferenceModel
import com.example.distilbert.ml.LoadDatasetClient
import com.example.distilbert.ml.QaAnswer
import com.example.distilbert.ml.QaClient
import com.google.ai.client.generativeai.GenerativeModel
import com.google.android.material.snackbar.Snackbar
import com.google.android.material.textfield.TextInputEditText
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.Locale

/**
 * Activity for doing Q&A on a specific dataset
 */
class QaActivity : AppCompatActivity() {
    private var questionEditText: TextInputEditText? = null
    private var contentTextView: TextView? = null
    private var textToSpeech: TextToSpeech? = null
    private var questionAnswered = false
    private var content: String? = null
    private var handler: Handler? = null
    private var qaClient: QaClient? = null
    var generativeModel = GenerativeModel( /* modelName */"gemini-pro", API_KEY)
    private lateinit var chatViewModel: ChatViewModel
    //var model = GenerativeModelFutures.from(gm)
    override fun onCreate(savedInstanceState: Bundle?) {
        Log.v(TAG, "onCreate")
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_qa)

        val askButtonGemma = findViewById<TextView>(R.id.ask_button_gemma)
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val inferenceModel = InferenceModel.getInstance(this@QaActivity)
                // Notify the UI that the model has finished loading
                withContext(Dispatchers.Main) {
                    askButtonGemma.visibility = View.VISIBLE
                    findViewById<ProgressBar>(R.id.progress_bar).visibility = View.GONE
                    chatViewModel = ChatViewModel(inferenceModel)

                    // Set listener
                    askButtonGemma.setOnClickListener {
                        findViewById<TextView>(R.id.gemini_result).text = ""
                        findViewById<ProgressBar>(R.id.progress_bar).visibility = View.VISIBLE
                        chatViewModel.resetText()
                        var question = questionEditText!!.text.toString()
                        question = question.trim { it <= ' ' }
                        if (question.isEmpty()) {
                            questionEditText!!.setText(question)
                            return@setOnClickListener
                        }

                        // Append question mark '?' if not ended with '?'.stion
                        // This aligns with question format that trains the model.
                        if (!question.endsWith("?")) {
                            question += '?'
                        }
                        val questionToAsk = question
                        Log.v("Question", "$content $questionToAsk")
                        lifecycleScope.launch(Dispatchers.IO) {
                            val completeQuery = "Here is the context: $content\n" +
                                    "Using the relevant information from the context \n" +
                                    "and integrating it with your knowledge,\n" +
                                    "provide an answer as helpful and informative bot that answers questions using text from the reference passage included below to the question: $questionToAsk.\n" +
                                    "If the context doesn't provide\n" +
                                    "any relevant information answer with \n" +
                                    "[I couldn't find a good match in my\n" +
                                    "knowledge base for your question, \n" +
                                    "hence I answer based on my own knowledge] \\\n" +
                                    "ANSWER:"
                            val markQuery = "You are a helpful and informative bot that answers questions using text from the reference passage included below. Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. However, you are talking to a non-technical audience, so be sure to break down complicated concepts and strike a friendly and conversational tone. If the passage is irrelevant to the answer, you may ignore it.\n" +
                                    "\n" +
                                    "\n" +
                                    "QUESTION: $questionToAsk\n" +
                                    "PASSAGE: $content\n" +
                                    "ANSWER:"
                            val simpleQnA = "PASSAGE: $content\n" + "QUESTION: $questionToAsk\n" + "ANSWER:"
                            chatViewModel.sendMessage(simpleQnA)
                        }
                    }

                    chatViewModel.gemmaOutput.observe(this@QaActivity, Observer { gemmaText ->

                        findViewById<TextView>(R.id.gemini_result).text = ""
                        findViewById<TextView>(R.id.gemini_result).text = gemmaText
                    })

                    chatViewModel.outputDone.observe(this@QaActivity) { done ->
                        if (done) {
                            findViewById<ProgressBar>(R.id.progress_bar).visibility = View.GONE
                            /*if (textToSpeech != null) {
                                textToSpeech!!.speak(chatViewModel.gemmaOutput.value, TextToSpeech.QUEUE_FLUSH, null, chatViewModel.gemmaOutput.value)
                            }*/
                        }
                    }
                }
            } catch (e: Exception) {
                Log.e("Error loading", e.toString())
            }
        }

        // Get content of the selected dataset.
        val datasetPosition = intent.getIntExtra(DATASET_POSITION_KEY, -1)
        val datasetClient = LoadDatasetClient(this)

        // Show the dataset title.
        val titleText = findViewById<TextView>(R.id.title_text)
        titleText.text = datasetClient.titles[datasetPosition]

        // Show the text content of the selected dataset.
        content = datasetClient.getContent(datasetPosition)
        contentTextView = findViewById(R.id.content_text)
        contentTextView?.setText(content)
        contentTextView?.setMovementMethod(ScrollingMovementMethod())

        // Setup question suggestion list.
        val questionSuggestionsView = findViewById<RecyclerView>(R.id.suggestion_list)
        val adapter = QuestionAdapter(this, datasetClient.getQuestions(datasetPosition))
        adapter.setOnQuestionSelectListener { question: String -> answerQuestion(question) }
        questionSuggestionsView.adapter = adapter
        val layoutManager = LinearLayoutManager(this, LinearLayoutManager.VERTICAL, false)
        questionSuggestionsView.layoutManager = layoutManager

        // Setup ask button for Bert.
        val askButtonBert = findViewById<TextView>(R.id.ask_button_bert)
        askButtonBert.setOnClickListener { view: View? ->
            findViewById<TextView>(R.id.gemini_result).text = ""
            answerQuestion(
                questionEditText!!.text.toString()
            )
        }

        // Setup ask button for Gemini.
        val askButtonGemini = findViewById<TextView>(R.id.ask_button_gemini)
        askButtonGemini.setOnClickListener { view: View? ->
            if (API_KEY != "") {
                findViewById<TextView>(R.id.gemini_result).text = ""
                val runningSnackbar =
                    Snackbar.make(contentTextView!!, "Connecting to Gemini Pro...", Snackbar.LENGTH_SHORT)
                runningSnackbar.show()

                var question = questionEditText!!.text.toString()
                question = question.trim { it <= ' ' }
                if (question.isEmpty()) {
                    questionEditText!!.setText(question)
                    return@setOnClickListener
                }

                // Append question mark '?' if not ended with '?'.stion
                // This aligns with question format that trains the model.
                if (!question.endsWith("?")) {
                    question += '?'
                }
                val questionToAsk = question
                lifecycleScope.launch {
                    val markQuery = "You are a helpful and informative bot that answers questions using text from the reference passage included below. " +
                            "Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. " +
                            "However, you are talking to a non-technical audience, so be sure to break down complicated concepts and strike a friendly and conversational tone. " +
                            "If the passage is irrelevant to the answer, you may ignore it.\n" +
                            "\n" +
                            "\n" +
                            "PASSAGE: $content\n" +
                            "QUESTION: $questionToAsk\n" +
                            "ANSWER:"
                    Log.v("Question", "$markQuery")
                    val returnedText = generativeModel.generateContent(markQuery)

                    returnedText.text?.let { Log.v("Answer", it) }

                    findViewById<TextView>(R.id.gemini_result).text = returnedText.text ?: ""
                    if (textToSpeech != null) {
                        textToSpeech!!.speak(returnedText.text ?: "", TextToSpeech.QUEUE_FLUSH, null, returnedText.text ?: "")
                    }
                }
            } else {
                val runningSnackbar =
                    Snackbar.make(contentTextView!!, "Get an API key and use a VPN if your country has no access to Gemini.", Snackbar.LENGTH_INDEFINITE)
                runningSnackbar.show()
            }
        }

        // Setup text edit where users can input their question.
        questionEditText = findViewById(R.id.question_edit_text)
        questionEditText?.setOnFocusChangeListener(
            OnFocusChangeListener { view: View?, hasFocus: Boolean ->
                // If we already answer current question, clear the question so that user can input a new
                // one.
                if (hasFocus && questionAnswered) {
                    questionEditText?.setText(null)
                }
            })
        questionEditText?.addTextChangedListener(
            object : TextWatcher {
                override fun beforeTextChanged(
                    charSequence: CharSequence,
                    i: Int,
                    i1: Int,
                    i2: Int
                ) {
                }

                override fun onTextChanged(charSequence: CharSequence, i: Int, i1: Int, i2: Int) {
                    // Only allow clicking Ask button if there is a question.
                    val shouldAskButtonActive = !charSequence.toString().isEmpty()
                    //askButton.setClickable(shouldAskButtonActive);
                    //askButton.setImageResource(shouldAskButtonActive ? R.drawable.ic_ask_active : R.drawable.ic_ask_inactive);
                }

                override fun afterTextChanged(editable: Editable) {}
            })
        questionEditText?.setOnKeyListener(
            View.OnKeyListener { v: View?, keyCode: Int, event: KeyEvent ->
                if (event.action == KeyEvent.ACTION_UP && keyCode == KeyEvent.KEYCODE_ENTER) {
                    answerQuestion(questionEditText?.getText().toString())
                }
                false
            })

        // Setup QA client to and background thread to run inference.
        val handlerThread = HandlerThread("QAClient")
        handlerThread.start()
        handler = Handler(handlerThread.looper)
        qaClient = QaClient(this)
    }

    override fun onStart() {
        Log.v(TAG, "onStart")
        super.onStart()
        handler!!.post {
            qaClient!!.loadModel()
            qaClient!!.loadDictionary()
        }
        textToSpeech = TextToSpeech(
            this
        ) { status: Int ->
            if (status == TextToSpeech.SUCCESS) {
                textToSpeech!!.setLanguage(Locale.US)
            } else {
                textToSpeech = null
            }
        }
    }

    override fun onStop() {
        Log.v(TAG, "onStop")
        super.onStop()
        handler!!.post { qaClient!!.unload() }
        if (textToSpeech != null) {
            textToSpeech!!.stop()
            textToSpeech!!.shutdown()
        }
    }

    private fun answerQuestion(question: String) {
        var question = question
        question = question.trim { it <= ' ' }
        if (question.isEmpty()) {
            questionEditText!!.setText(question)
            return
        }

        // Append question mark '?' if not ended with '?'.
        // This aligns with question format that trains the model.
        if (!question.endsWith("?")) {
            question += '?'
        }
        val questionToAsk = question
        questionEditText!!.setText(questionToAsk)

        // Delete all pending tasks.
        handler!!.removeCallbacksAndMessages(null)

        // Hide keyboard and dismiss focus on text edit.
        val imm = getSystemService(INPUT_METHOD_SERVICE) as InputMethodManager
        imm.hideSoftInputFromWindow(window.decorView.windowToken, 0)
        val focusView = currentFocus
        focusView?.clearFocus()

        // Reset content text view
        contentTextView!!.text = content
        questionAnswered = false
        val runningSnackbar =
            Snackbar.make(contentTextView!!, "Looking up answer...", Snackbar.LENGTH_INDEFINITE)
        runningSnackbar.show()

        // Run TF Lite model to get the answer.
        handler!!.post {
            val beforeTime = System.currentTimeMillis()
            val answers = qaClient!!.predict(questionToAsk, content)
            val afterTime = System.currentTimeMillis()
            val totalSeconds = (afterTime - beforeTime) / 1000.0
            if (!answers.isEmpty()) {
                // Get the top answer
                val topAnswer = answers[0]
                // Show the answer.
                runOnUiThread {
                    runningSnackbar.dismiss()
                    presentAnswer(topAnswer)
                    var displayMessage = "Top answer was successfully highlighted."
                    if (DISPLAY_RUNNING_TIME) {
                        displayMessage = String.format("%s %.3fs.", displayMessage, totalSeconds)
                    }
                    Snackbar.make(contentTextView!!, displayMessage, Snackbar.LENGTH_LONG).show()
                    questionAnswered = true
                }
            }
        }
    }

    private fun presentAnswer(answer: QaAnswer) {
        // Highlight answer.
        val spanText: Spannable = SpannableString(content)
        val offset = content!!.indexOf(answer.text, 0)
        if (offset >= 0) {
            spanText.setSpan(
                BackgroundColorSpan(getColor(R.color.secondaryColor)),
                offset,
                offset + answer.text.length,
                Spannable.SPAN_EXCLUSIVE_EXCLUSIVE
            )
        }
        contentTextView!!.text = spanText

        // Use TTS to speak out the answer.
        if (textToSpeech != null) {
            textToSpeech!!.speak(answer.text, TextToSpeech.QUEUE_FLUSH, null, answer.text)
        }
    }

    companion object {
        private const val DATASET_POSITION_KEY = "DATASET_POSITION"
        private const val TAG = "QaActivity"
        private const val API_KEY = ""
        private const val DISPLAY_RUNNING_TIME = false
        @JvmStatic
        fun newInstance(context: Context?, datasetPosition: Int): Intent {
            val intent = Intent(context, QaActivity::class.java)
            intent.putExtra(DATASET_POSITION_KEY, datasetPosition)
            return intent
        }
    }
}
