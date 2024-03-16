package com.example.distilbert.gemma

import android.util.Log
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.collectIndexed
import kotlinx.coroutines.launch

class ChatViewModel(
    private val inferenceModel: InferenceModel
) : ViewModel() {

    // `GemmaUiState()` is optimized for the Gemma model.
    // Replace `GemmaUiState` with `ChatUiState()` if you're using a different model
    private val _uiState: MutableStateFlow<GemmaUiState> = MutableStateFlow(GemmaUiState())
    val uiState: StateFlow<UiState> = _uiState.asStateFlow()

    private val _gemmaOutput: MutableLiveData<String> = MutableLiveData("")
    val gemmaOutput: LiveData<String> = _gemmaOutput

    private val _outputDone: MutableLiveData<Boolean> = MutableLiveData(false)
    val outputDone: LiveData<Boolean> = _outputDone

    private val _textInputEnabled: MutableStateFlow<Boolean> =
        MutableStateFlow(true)
    val isTextInputEnabled: StateFlow<Boolean> =
        _textInputEnabled.asStateFlow()

    fun sendMessage(userMessage: String) {
        viewModelScope.launch(Dispatchers.IO) {
            _uiState.value.addMessage(userMessage, USER_PREFIX)
            var currentMessageId: String? = _uiState.value.createLoadingMessage()
            try {
                val fullPrompt = _uiState.value.fullPrompt
                Log.v("Full_prompt", fullPrompt)
                inferenceModel.generateResponseAsync(fullPrompt)
                inferenceModel.partialResults
                    .collectIndexed { index, (partialResult, done) ->
                        currentMessageId?.let {
                            _outputDone.postValue(false)
                            Log.v("Answer", partialResult)

                            _gemmaOutput.postValue("${_gemmaOutput.value}$partialResult") // Concatenate the new text

                            /*if (index == 0) {
                                _uiState.value.appendFirstMessage(it, partialResult)
                            } else {
                                _uiState.value.appendMessage(it, partialResult, done)
                            }*/
                            _uiState.value.appendMessage(it, partialResult, done)
                            if (done) {
                                currentMessageId = null
                                // Re-enable text input
                                _outputDone.postValue(true)
                            }
                        }
                    }
            } catch (e: Exception) {
                _uiState.value.addMessage(e.localizedMessage ?: "Unknown Error", MODEL_PREFIX)
            }
        }
    }

    fun resetText() {
        _gemmaOutput.postValue("")
    }


}
