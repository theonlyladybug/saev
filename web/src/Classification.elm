-- Add https://tailwindcss.com/blog/automatic-class-sorting-with-prettier to elm-format.


module Classification exposing (..)

import Array
import Browser
import Browser.Navigation
import Dict
import Gradio
import Html
import Html.Attributes
import Html.Events
import Http
import Json.Decode as D
import Json.Encode as E
import Random
import Requests
import Set
import Url
import Url.Builder
import Url.Parser exposing ((</>), (<?>))
import Url.Parser.Query


main =
    Browser.application
        { init = init
        , view = view
        , update = update
        , subscriptions = \model -> Sub.none
        , onUrlRequest = onUrlRequest
        , onUrlChange = onUrlChange
        }



-- MESSAGE


type Msg
    = NoOp
    | SetUrl Int
    | SetExample Int
    | GetRandomExample
    | HoverPatch Int
    | ResetHoveredPatch
    | ToggleSelectedPatch Int
    | SetSlider Int String
    | ExamineClass Int
      -- API responses
    | GotInputExample Requests.Id (Result Gradio.Error Example)
    | GotOriginalPredictions Requests.Id (Result Gradio.Error (List Float))
    | GotModifiedPredictions Requests.Id (Result Gradio.Error (List Float))
    | GotSaeExamples Requests.Id (Result Gradio.Error (List SaeExampleResult))
    | GotClassExample Requests.Id (Result Gradio.Error Example)



-- MODEL


type alias Model =
    { -- Browser
      key : Browser.Navigation.Key
    , inputExampleIndex : Int
    , inputExample : Requests.Requested Example
    , hoveredPatchIndex : Maybe Int
    , selectedPatchIndices : Set.Set Int
    , saeExamples : Requests.Requested (List SaeExample)
    , sliders : Dict.Dict Int Float
    , examinedClass : ExaminedClass

    -- ML stuff
    , originalPredictions : Requests.Requested (List Float)
    , modifiedPredictions : Requests.Requested (List Float)

    -- APIs
    , gradio : Gradio.Config
    , inputExampleRequestId : Requests.Id
    , originalPredictionsRequestId : Requests.Id
    , modifiedPredictionsRequestId : Requests.Id
    , classExampleRequestId : Requests.Id
    , saeExamplesRequestId : Requests.Id
    }


type ExaminedClass
    = NotExamining
    | Examining
        { class : Int
        , examples : Requests.Requested (List Example)
        }


type alias Example =
    { url : String
    , class : Int
    }


type alias SaeExample =
    { latent : Int
    , urls : List String
    }


init : () -> Url.Url -> Browser.Navigation.Key -> ( Model, Cmd Msg )
init _ url key =
    let
        index =
            case Url.Parser.parse urlParser url of
                Just parsed ->
                    case parsed.example of
                        Just i ->
                            i

                        Nothing ->
                            0

                Nothing ->
                    0

        model =
            { -- Browser
              key = key
            , inputExampleIndex = index
            , inputExample = Requests.Loading
            , hoveredPatchIndex = Nothing
            , selectedPatchIndices = Set.empty
            , saeExamples = Requests.Initial
            , sliders = Dict.empty
            , examinedClass = NotExamining

            -- ML
            , originalPredictions = Requests.Loading
            , modifiedPredictions = Requests.Initial

            -- APIs
            , inputExampleRequestId = Requests.init
            , originalPredictionsRequestId = Requests.init
            , modifiedPredictionsRequestId = Requests.init
            , classExampleRequestId = Requests.init
            , saeExamplesRequestId = Requests.init
            , gradio =
                { host = "https://samuelstevens-saev-image-classification.hf.space" }
            }
    in
    ( model
    , Cmd.batch
        [ getInputExample model.gradio model.inputExampleRequestId index
        , getOriginalPredictions model.gradio model.originalPredictionsRequestId index
        ]
    )



-- UPDATE


update : Msg -> Model -> ( Model, Cmd Msg )
update msg model =
    case msg of
        NoOp ->
            ( model, Cmd.none )

        SetUrl i ->
            let
                url =
                    Url.Builder.relative [] [ Url.Builder.int "example" i ]
            in
            ( model, Browser.Navigation.pushUrl model.key url )

        SetExample i ->
            let
                example =
                    modBy nImages i

                inputExampleNextId =
                    Requests.next model.inputExampleRequestId

                originalPredictionsNextId =
                    Requests.next model.originalPredictionsRequestId
            in
            ( { model
                | inputExampleIndex = example
                , inputExample = Requests.Loading
                , selectedPatchIndices = Set.empty
                , saeExamples = Requests.Initial
                , originalPredictions = Requests.Loading
                , modifiedPredictions = Requests.Initial
                , inputExampleRequestId = inputExampleNextId
                , originalPredictionsRequestId = originalPredictionsNextId
              }
            , Cmd.batch
                [ getInputExample model.gradio inputExampleNextId example
                , getOriginalPredictions model.gradio originalPredictionsNextId example
                ]
            )

        GetRandomExample ->
            ( { model | examinedClass = NotExamining }, Random.generate SetUrl randomExample )

        HoverPatch i ->
            ( { model | hoveredPatchIndex = Just i }, Cmd.none )

        ResetHoveredPatch ->
            ( { model | hoveredPatchIndex = Nothing }, Cmd.none )

        GotInputExample id result ->
            if Requests.isStale id model.inputExampleRequestId then
                ( model, Cmd.none )

            else
                case result of
                    Ok example ->
                        ( { model | inputExample = Requests.Loaded example }, Cmd.none )

                    Err err ->
                        ( { model | inputExample = Requests.Failed (explainGradioError err) }, Cmd.none )

        GotOriginalPredictions id result ->
            if Requests.isStale id model.originalPredictionsRequestId then
                ( model, Cmd.none )

            else
                case result of
                    Ok preds ->
                        ( { model | originalPredictions = Requests.Loaded preds }, Cmd.none )

                    Err err ->
                        ( { model
                            | originalPredictions = Requests.Failed (explainGradioError err)
                          }
                        , Cmd.none
                        )

        GotModifiedPredictions id result ->
            if Requests.isStale id model.modifiedPredictionsRequestId then
                ( model, Cmd.none )

            else
                case result of
                    Ok modified ->
                        ( { model | modifiedPredictions = Requests.Loaded modified }, Cmd.none )

                    Err err ->
                        ( { model
                            | modifiedPredictions = Requests.Failed (explainGradioError err)
                          }
                        , Cmd.none
                        )

        ToggleSelectedPatch i ->
            let
                patchIndices =
                    if Set.member i model.selectedPatchIndices then
                        Set.remove i model.selectedPatchIndices

                    else
                        Set.insert i model.selectedPatchIndices

                saeExamplesNextId =
                    Requests.next model.saeExamplesRequestId
            in
            ( { model
                | selectedPatchIndices = patchIndices
                , saeExamplesRequestId = saeExamplesNextId
                , saeExamples = Requests.Loading
              }
            , getSaeExamples model.gradio saeExamplesNextId model.inputExampleIndex patchIndices
            )

        GotSaeExamples id result ->
            if Requests.isStale id model.saeExamplesRequestId then
                ( model, Cmd.none )

            else
                case result of
                    Ok results ->
                        let
                            latents =
                                parseSaeExampleResults results

                            sliders =
                                latents
                                    |> List.map (\latent -> ( latent.latent, 0.0 ))
                                    |> Dict.fromList
                        in
                        ( { model
                            | saeExamples = Requests.Loaded latents
                            , sliders = sliders
                          }
                        , Cmd.none
                        )

                    Err err ->
                        ( { model | saeExamples = Requests.Failed (explainGradioError err) }, Cmd.none )

        SetSlider i str ->
            case String.toFloat str of
                Just f ->
                    let
                        sliders =
                            Dict.insert i f model.sliders

                        modifiedPredictionsNextId =
                            Requests.next model.modifiedPredictionsRequestId
                    in
                    ( { model
                        | sliders = sliders
                        , modifiedPredictionsRequestId = modifiedPredictionsNextId
                        , modifiedPredictions = Requests.Loading
                      }
                    , getModifiedPredictions model.gradio
                        modifiedPredictionsNextId
                        model.inputExampleIndex
                        model.selectedPatchIndices
                        sliders
                    )

                Nothing ->
                    ( model, Cmd.none )

        ExamineClass class ->
            let
                -- We actually want all the requests to have the same ID because they all represent the same "logical" request.
                id =
                    Requests.next model.classExampleRequestId
            in
            ( { model
                | examinedClass = Examining { class = class, examples = Requests.Loading }
                , classExampleRequestId = id
              }
            , getRandomClassExample model.gradio id class
                |> List.repeat 9
                |> Cmd.batch
            )

        GotClassExample id result ->
            if Requests.isStale id model.classExampleRequestId then
                ( model, Cmd.none )

            else
                case ( result, model.examinedClass ) of
                    ( Ok example, NotExamining ) ->
                        ( { model
                            | examinedClass =
                                Examining
                                    { class = example.class
                                    , examples = Requests.Loaded [ example ]
                                    }
                          }
                        , Cmd.none
                        )

                    ( Ok example, Examining { class, examples } ) ->
                        case examples of
                            Requests.Initial ->
                                let
                                    newExamples =
                                        if example.class == class then
                                            [ example ]

                                        else
                                            []
                                in
                                ( { model
                                    | examinedClass =
                                        Examining
                                            { class = class
                                            , examples = Requests.Loaded newExamples
                                            }
                                  }
                                , Cmd.none
                                )

                            Requests.Loading ->
                                let
                                    newExamples =
                                        if example.class == class then
                                            [ example ]

                                        else
                                            []
                                in
                                ( { model
                                    | examinedClass =
                                        Examining
                                            { class = class
                                            , examples = Requests.Loaded newExamples
                                            }
                                  }
                                , Cmd.none
                                )

                            Requests.Loaded examplesLoaded ->
                                let
                                    newExamples =
                                        example
                                            :: examplesLoaded
                                            |> List.filter (\ex -> ex.class == class)
                                in
                                ( { model
                                    | examinedClass = Examining { class = class, examples = Requests.Loaded newExamples }
                                  }
                                , Cmd.none
                                )

                            Requests.Failed _ ->
                                let
                                    newExamples =
                                        if example.class == class then
                                            [ example ]

                                        else
                                            []
                                in
                                ( { model
                                    | examinedClass =
                                        Examining
                                            { class = class
                                            , examples = Requests.Loaded newExamples
                                            }
                                  }
                                , Cmd.none
                                )

                    ( Err err, _ ) ->
                        ( { model
                            | examinedClass =
                                Examining
                                    { class = -1
                                    , examples = Requests.Failed (explainGradioError err)
                                    }
                          }
                        , Cmd.none
                        )


onUrlRequest : Browser.UrlRequest -> Msg
onUrlRequest request =
    NoOp


onUrlChange : Url.Url -> Msg
onUrlChange url =
    case Url.Parser.parse urlParser url of
        Just parsed ->
            case parsed.example of
                Just i ->
                    SetExample i

                Nothing ->
                    SetExample 0

        Nothing ->
            SetExample 0


type alias QueryParams =
    { example : Maybe Int
    }


urlParser : Url.Parser.Parser (QueryParams -> a) a
urlParser =
    -- Need to change this when I deploy it.
    Url.Parser.s "saev"
        </> Url.Parser.s "demos"
        </> Url.Parser.s "classification"
        <?> Url.Parser.Query.int "example"
        |> Url.Parser.map QueryParams



-- API


explainGradioError : Gradio.Error -> String
explainGradioError err =
    case err of
        Gradio.NetworkError msg ->
            "Network error: " ++ msg

        Gradio.JsonError msg ->
            "Error decoding JSON: " ++ msg

        Gradio.ParsingError msg ->
            "Error parsing API response: " ++ msg

        Gradio.ApiError msg ->
            "Error in the API: " ++ msg


getInputExample : Gradio.Config -> Requests.Id -> Int -> Cmd Msg
getInputExample cfg id img =
    Gradio.get cfg
        "get-image"
        [ E.int img ]
        (D.map2 Example
            (D.index 0 imgUrlDecoder)
            (D.index 1 D.int)
        )
        (GotInputExample id)


getOriginalPredictions : Gradio.Config -> Requests.Id -> Int -> Cmd Msg
getOriginalPredictions cfg id img =
    Gradio.get cfg
        "get-preds"
        [ E.int img ]
        (Gradio.decodeOne labelsDecoder)
        (GotOriginalPredictions id)


getModifiedPredictions : Gradio.Config -> Requests.Id -> Int -> Set.Set Int -> Dict.Dict Int Float -> Cmd Msg
getModifiedPredictions cfg id img patches sliders =
    let
        ( latents, values ) =
            sliders
                |> Dict.toList
                |> List.unzip
                |> Tuple.mapBoth Array.fromList Array.fromList

        args =
            [ E.int img
            , E.list E.int (Set.toList patches)
            , E.int (Array.get 0 latents |> Maybe.withDefault -1)
            , E.int (Array.get 1 latents |> Maybe.withDefault -1)
            , E.int (Array.get 2 latents |> Maybe.withDefault -1)
            , E.float (Array.get 0 values |> Maybe.withDefault 0.0)
            , E.float (Array.get 1 values |> Maybe.withDefault 0.0)
            , E.float (Array.get 2 values |> Maybe.withDefault 0.0)
            ]
    in
    Gradio.get cfg
        "get-modified"
        args
        (D.list labelsDecoder
            |> D.map List.head
            |> D.map (Maybe.withDefault [])
        )
        (GotModifiedPredictions id)


labelsDecoder : D.Decoder (List Float)
labelsDecoder =
    D.field "confidences"
        (D.list
            (D.map2 Tuple.pair
                (D.field "confidence" D.float)
                (D.field "label" D.int)
            )
            |> D.andThen (List.sortBy Tuple.second >> List.map Tuple.first >> D.succeed)
        )


getSaeExamples : Gradio.Config -> Requests.Id -> Int -> Set.Set Int -> Cmd Msg
getSaeExamples cfg id img patches =
    Gradio.get cfg
        "get-sae-examples"
        [ E.int img
        , Set.toList patches |> E.list E.int
        ]
        (D.list saeExampleResultDecoder)
        (GotSaeExamples id)


type SaeExampleResult
    = SaeExampleMissing
    | SaeExampleUrl String
    | SaeExampleLatent Int


saeExampleResultDecoder : D.Decoder SaeExampleResult
saeExampleResultDecoder =
    D.oneOf
        [ imgUrlDecoder |> D.map SaeExampleUrl
        , D.int |> D.map SaeExampleLatent
        , D.null SaeExampleMissing
        ]


imgUrlDecoder : D.Decoder String
imgUrlDecoder =
    D.field "url" D.string
        |> D.map (String.replace "gradio_api/gradio_api" "gradio_api")
        |> D.map (String.replace "gra/gradio_api" "gradio_api")
        |> D.map (String.replace "gradio_ap/gradio_api" "gradio_api")
        |> D.map (String.replace "gradio_api/ca/gradio_api" "gradio_api")
        |> D.map (String.replace "gradio_api/call/gradio_api" "gradio_api")


getRandomClassExample : Gradio.Config -> Requests.Id -> Int -> Cmd Msg
getRandomClassExample cfg id class =
    Gradio.get cfg
        "get-random-class-image"
        [ E.int class ]
        (D.map2 Example
            (D.index 0 imgUrlDecoder)
            (D.succeed class)
        )
        (GotClassExample id)



-- VIEW


view : Model -> Browser.Document Msg
view model =
    let
        callToAction =
            case model.saeExamples of
                Requests.Loaded _ ->
                    "Drag one or more sliders below."

                _ ->
                    ""
    in
    { title = "Image Classification"
    , body =
        [ Html.header [] []
        , Html.main_
            [ Html.Attributes.class "w-full min-h-screen p-1 md:p-2 lg:p-4 bg-gray-50 space-y-4" ]
            [ Html.div
                [ Html.Attributes.class "border border-gray-200 bg-white rounded-lg p-4 space-y-4" ]
                [ Html.div
                    [ Html.Attributes.class "flex flex-row gap-2 items-stretch" ]
                    [ viewInputExample model
                    , viewProbs "Prediction"
                        "Wait just a second."
                        model.originalPredictions
                    , viewProbs "After Modification"
                        callToAction
                        model.modifiedPredictions
                    ]
                , Html.div [ Html.Attributes.class "border-t border-gray-200" ] []
                , Html.div
                    [ Html.Attributes.class "flex flex-row justify-between" ]
                    [ viewSaeExamples model.saeExamples model.sliders
                    , viewClassExamples model.examinedClass
                    ]
                ]
            ]
        ]
    }


viewInputExample : Model -> Html.Html Msg
viewInputExample model =
    case model.inputExample of
        Requests.Initial ->
            Html.p
                []
                [ Html.text "Initial state. You shouldn't see this for long..." ]

        Requests.Loading ->
            Html.p
                []
                [ Html.text "Loading example..." ]

        Requests.Loaded example ->
            Html.div
                [ Html.Attributes.class "w-[224px] md:w-[336px]" ]
                [ viewGriddedImage
                    model.hoveredPatchIndex
                    model.selectedPatchIndices
                    example.url
                , Html.div
                    [ Html.Attributes.class "flex flex-row gap-2" ]
                    [ viewButton (SetUrl (model.inputExampleIndex - 1)) "Previous"
                    , viewButton GetRandomExample "Random"
                    , viewButton (SetUrl (model.inputExampleIndex + 1)) "Next"
                    ]
                , Html.div
                    [ Html.Attributes.class "py-1 md:py-2 lg:py-4" ]
                    [ Html.p
                        [ Html.Attributes.class "font-bold text-gray-800" ]
                        [ Html.text
                            (viewClass example.class
                                ++ " (Example #"
                                ++ String.fromInt model.inputExampleIndex
                                ++ ")"
                            )
                        ]
                    ]
                ]

        Requests.Failed err ->
            viewErr err


viewGriddedImage : Maybe Int -> Set.Set Int -> String -> Html.Html Msg
viewGriddedImage hovered selected url =
    Html.div
        [ Html.Attributes.class "relative inline-block" ]
        [ Html.div
            [ Html.Attributes.class "absolute grid grid-rows-[repeat(14,_16px)] grid-cols-[repeat(14,_16px)] md:grid-rows-[repeat(14,_24px)] md:grid-cols-[repeat(14,_24px)]" ]
            (List.map
                (viewGridCell hovered selected)
                (List.range 0 195)
            )
        , Html.img
            [ Html.Attributes.class "block w-[224px] h-[224px] md:w-[336px] md:h-[336px]"
            , Html.Attributes.src url
            ]
            []
        ]


viewGridCell : Maybe Int -> Set.Set Int -> Int -> Html.Html Msg
viewGridCell hovered selected self =
    let
        class =
            (case hovered of
                Just h ->
                    if h == self then
                        "border-2 border-rose-600 border-dashed "

                    else
                        ""

                Nothing ->
                    ""
            )
                ++ (if Set.member self selected then
                        "bg-rose-600/50 "

                    else
                        ""
                   )
    in
    Html.div
        [ Html.Attributes.class "w-[16px] h-[16px] md:w-[24px] md:h-[24px]"
        , Html.Attributes.class class
        , Html.Events.onMouseEnter (HoverPatch self)
        , Html.Events.onMouseLeave ResetHoveredPatch
        , Html.Events.onClick (ToggleSelectedPatch self)
        ]
        []


viewProbs : String -> String -> Requests.Requested (List Float) -> Html.Html Msg
viewProbs title callToAction loadingProbs =
    let
        content =
            case loadingProbs of
                Requests.Initial ->
                    Html.p
                        [ Html.Attributes.class "italic text-gray-600" ]
                        [ Html.text callToAction ]

                Requests.Loading ->
                    Html.p
                        []
                        [ Html.text "Loading..." ]

                Requests.Loaded probs ->
                    let
                        top =
                            probs
                                |> List.indexedMap Tuple.pair
                                |> List.sortBy Tuple.second
                                |> List.reverse
                                |> List.take 6
                    in
                    Html.div
                        []
                        (List.map (uncurry viewProb) top)

                Requests.Failed err ->
                    viewErr err
    in
    Html.div
        [ Html.Attributes.class "pt-1 md:pt-2 lg:pt-4 flex-1" ]
        [ Html.h3
            [ Html.Attributes.class "font-bold text-gray-800" ]
            [ Html.text title ]
        , content
        ]


viewProb : Int -> Float -> Html.Html Msg
viewProb target prob =
    Html.div
        [ Html.Attributes.class "cursor-pointer hover:underline"
        , Html.Events.onClick (ExamineClass target)
        ]
        [ Html.meter
            [ Html.Attributes.class "inline-block w-full"
            , Html.Attributes.min "0"
            , Html.Attributes.max "100"
            , Html.Attributes.value (prob * 100 |> String.fromFloat)
            ]
            []
        , Html.div
            [ Html.Attributes.class "flex justify-between"
            ]
            [ Html.span
                [ Html.Attributes.class "text-gray-700" ]
                [ Html.text (viewClass target) ]
            , Html.span
                [ Html.Attributes.class "text-gray-900" ]
                [ Html.text ((prob * 100 |> round |> String.fromInt) ++ "%") ]
            ]
        ]


viewSaeExamples : Requests.Requested (List SaeExample) -> Dict.Dict Int Float -> Html.Html Msg
viewSaeExamples examplesLoading values =
    case examplesLoading of
        Requests.Initial ->
            Html.p
                [ Html.Attributes.class "italic text-gray-600" ]
                [ Html.text "Click on the image above to find similar image patches using a sparse autoencoder (SAE)." ]

        Requests.Loading ->
            Html.p []
                [ Html.text "Loading similar patches..." ]

        Requests.Loaded examples ->
            Html.div [ Html.Attributes.class "p-1 md:px-2 lg:px-4" ]
                ([ Html.h3
                    [ Html.Attributes.class "font-bold" ]
                    [ Html.text "Similar Examples" ]
                 , Html.p []
                    [ Html.span [ Html.Attributes.class "bg-rose-600 text-white p-1 rounded" ] [ Html.text "These patches" ]
                    , Html.text " above are like "
                    , Html.span [ Html.Attributes.class "bg-rose-600 text-white p-1 rounded" ] [ Html.text "these patches" ]
                    , Html.text " below. (Not what you expected? Add more patches and get a larger "
                    , Html.a [ Html.Attributes.href "https://simple.wikipedia.org/wiki/Sampling_(statistics)", Html.Attributes.class "text-blue-500 underline" ] [ Html.text "sample size" ]
                    , Html.text ")"
                    ]
                 ]
                    ++ List.filterMap
                        (\example -> Maybe.map (\f -> viewSaeExample example f) (Dict.get example.latent values))
                        examples
                )

        Requests.Failed err ->
            viewErr err


viewSaeExample : SaeExample -> Float -> Html.Html Msg
viewSaeExample example value =
    Html.div
        [ Html.Attributes.class "flex flex-row gap-2 mt-2" ]
        (List.map viewImage example.urls
            ++ [ Html.div
                    [ Html.Attributes.class "flex flex-col gap-2" ]
                    [ Html.input
                        [ Html.Attributes.type_ "range"
                        , Html.Attributes.min "-20"
                        , Html.Attributes.max "20"
                        , Html.Attributes.value (String.fromFloat value)
                        , Html.Events.onInput (SetSlider example.latent)
                        ]
                        []
                    , Html.p
                        []
                        [ Html.text ("Latent 24K/" ++ String.fromInt example.latent ++ ": " ++ String.fromFloat value) ]
                    ]
               ]
        )


viewImage : String -> Html.Html Msg
viewImage url =
    Html.img
        [ Html.Attributes.src url
        , Html.Attributes.class "max-w-36 h-auto"
        ]
        []


viewErr : String -> Html.Html Msg
viewErr err =
    Html.div
        [ Html.Attributes.class "relative rounded-lg border border-red-200 bg-red-50 p-4" ]
        [ Html.button
            []
            []
        , Html.h3
            [ Html.Attributes.class "font-bold text-red-800" ]
            [ Html.text "Error" ]
        , Html.p
            [ Html.Attributes.class "text-red-700" ]
            [ Html.text err ]
        ]


viewClassExamples : ExaminedClass -> Html.Html Msg
viewClassExamples examined =
    case examined of
        NotExamining ->
            Html.div [] []

        Examining { class, examples } ->
            case examples of
                Requests.Initial ->
                    Html.div [] []

                Requests.Loading ->
                    Html.div [] []

                Requests.Loaded examplesLoaded ->
                    Html.div
                        [ Html.Attributes.class "" ]
                        [ Html.h3
                            [ Html.Attributes.class "font-bold" ]
                            [ class |> viewClass |> Html.text ]
                        , Html.div
                            [ Html.Attributes.class "max-w-xl grid grid-cols-3 gap-2" ]
                            (examplesLoaded |> List.reverse |> List.take 9 |> List.map (.url >> viewImage))
                        ]

                Requests.Failed err ->
                    viewErr err


viewButton : Msg -> String -> Html.Html Msg
viewButton onClick title =
    Html.button
        [ Html.Events.onClick onClick
        , Html.Attributes.class "flex-1 rounded-lg px-2 py-1 transition-colors"
        , Html.Attributes.class "border border-sky-300 hover:border-sky-400"
        , Html.Attributes.class "bg-sky-100 hover:bg-sky-200"
        , Html.Attributes.class "text-gray-700 hover:text-gray-900"
        , Html.Attributes.class "focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2"
        , Html.Attributes.class "active:bg-gray-300"
        ]
        [ Html.text title ]



-- HELPERS


uncurry : (a -> b -> c) -> ( a, b ) -> c
uncurry f ( a, b ) =
    f a b


parseSaeExampleResults : List SaeExampleResult -> List SaeExample
parseSaeExampleResults results =
    let
        ( _, _, parsed ) =
            parseSaeExampleResultsHelper results [] []
    in
    parsed


parseSaeExampleResultsHelper : List SaeExampleResult -> List (Maybe String) -> List SaeExample -> ( List SaeExampleResult, List (Maybe String), List SaeExample )
parseSaeExampleResultsHelper unparsed urls parsed =
    case unparsed of
        [] ->
            ( [], urls, parsed )

        (SaeExampleUrl url) :: rest ->
            parseSaeExampleResultsHelper
                rest
                (urls ++ [ Just url ])
                parsed

        (SaeExampleLatent latent) :: rest ->
            parseSaeExampleResultsHelper
                rest
                (List.drop nSaeExamplesPerLatent urls)
                (parsed
                    ++ [ { latent = latent
                         , urls =
                            urls
                                |> List.take nSaeExamplesPerLatent
                                |> List.filterMap identity
                         }
                       ]
                )

        SaeExampleMissing :: rest ->
            parseSaeExampleResultsHelper
                rest
                (urls ++ [ Nothing ])
                parsed


viewClass : Int -> String
viewClass class =
    Array.get class classNames
        |> Maybe.withDefault ("Unknown class: " ++ String.fromInt class)



-- CONSTANTS


nImages =
    5794


nSaeExamplesPerLatent =
    4


randomExample : Random.Generator Int
randomExample =
    Random.int 0 (nImages - 1)


classNames : Array.Array String
classNames =
    [ "Acadian_Flycatcher", "American_Crow", "American_Goldfinch", "American_Pipit", "American_Redstart", "American_Three_toed_Woodpecker", "Anna_Hummingbird", "Artic_Tern", "Baird_Sparrow", "Baltimore_Oriole", "Bank_Swallow", "Barn_Swallow", "Bay_breasted_Warbler", "Belted_Kingfisher", "Bewick_Wren", "Black_Tern", "Black_and_white_Warbler", "Black_billed_Cuckoo", "Black_capped_Vireo", "Black_footed_Albatross", "Black_throated_Blue_Warbler", "Black_throated_Sparrow", "Blue_Grosbeak", "Blue_Jay", "Blue_headed_Vireo", "Blue_winged_Warbler", "Boat_tailed_Grackle", "Bobolink", "Bohemian_Waxwing", "Brandt_Cormorant", "Brewer_Blackbird", "Brewer_Sparrow", "Bronzed_Cowbird", "Brown_Creeper", "Brown_Pelican", "Brown_Thrasher", "Cactus_Wren", "California_Gull", "Canada_Warbler", "Cape_Glossy_Starling", "Cape_May_Warbler", "Cardinal", "Carolina_Wren", "Caspian_Tern", "Cedar_Waxwing", "Cerulean_Warbler", "Chestnut_sided_Warbler", "Chipping_Sparrow", "Chuck_will_Widow", "Clark_Nutcracker", "Clay_colored_Sparrow", "Cliff_Swallow", "Common_Raven", "Common_Tern", "Common_Yellowthroat", "Crested_Auklet", "Dark_eyed_Junco", "Downy_Woodpecker", "Eared_Grebe", "Eastern_Towhee", "Elegant_Tern", "European_Goldfinch", "Evening_Grosbeak", "Field_Sparrow", "Fish_Crow", "Florida_Jay", "Forsters_Tern", "Fox_Sparrow", "Frigatebird", "Gadwall", "Geococcyx", "Glaucous_winged_Gull", "Golden_winged_Warbler", "Grasshopper_Sparrow", "Gray_Catbird", "Gray_Kingbird", "Gray_crowned_Rosy_Finch", "Great_Crested_Flycatcher", "Great_Grey_Shrike", "Green_Jay", "Green_Kingfisher", "Green_Violetear", "Green_tailed_Towhee", "Groove_billed_Ani", "Harris_Sparrow", "Heermann_Gull", "Henslow_Sparrow", "Herring_Gull", "Hooded_Merganser", "Hooded_Oriole", "Hooded_Warbler", "Horned_Grebe", "Horned_Lark", "Horned_Puffin", "House_Sparrow", "House_Wren", "Indigo_Bunting", "Ivory_Gull", "Kentucky_Warbler", "Laysan_Albatross", "Lazuli_Bunting", "Le_Conte_Sparrow", "Least_Auklet", "Least_Flycatcher", "Least_Tern", "Lincoln_Sparrow", "Loggerhead_Shrike", "Long_tailed_Jaeger", "Louisiana_Waterthrush", "Magnolia_Warbler", "Mallard", "Mangrove_Cuckoo", "Marsh_Wren", "Mockingbird", "Mourning_Warbler", "Myrtle_Warbler", "Nashville_Warbler", "Nelson_Sharp_tailed_Sparrow", "Nighthawk", "Northern_Flicker", "Northern_Fulmar", "Northern_Waterthrush", "Olive_sided_Flycatcher", "Orange_crowned_Warbler", "Orchard_Oriole", "Ovenbird", "Pacific_Loon", "Painted_Bunting", "Palm_Warbler", "Parakeet_Auklet", "Pelagic_Cormorant", "Philadelphia_Vireo", "Pied_Kingfisher", "Pied_billed_Grebe", "Pigeon_Guillemot", "Pileated_Woodpecker", "Pine_Grosbeak", "Pine_Warbler", "Pomarine_Jaeger", "Prairie_Warbler", "Prothonotary_Warbler", "Purple_Finch", "Red_bellied_Woodpecker", "Red_breasted_Merganser", "Red_cockaded_Woodpecker", "Red_eyed_Vireo", "Red_faced_Cormorant", "Red_headed_Woodpecker", "Red_legged_Kittiwake", "Red_winged_Blackbird", "Rhinoceros_Auklet", "Ring_billed_Gull", "Ringed_Kingfisher", "Rock_Wren", "Rose_breasted_Grosbeak", "Ruby_throated_Hummingbird", "Rufous_Hummingbird", "Rusty_Blackbird", "Sage_Thrasher", "Savannah_Sparrow", "Sayornis", "Scarlet_Tanager", "Scissor_tailed_Flycatcher", "Scott_Oriole", "Seaside_Sparrow", "Shiny_Cowbird", "Slaty_backed_Gull", "Song_Sparrow", "Sooty_Albatross", "Spotted_Catbird", "Summer_Tanager", "Swainson_Warbler", "Tennessee_Warbler", "Tree_Sparrow", "Tree_Swallow", "Tropical_Kingbird", "Vermilion_Flycatcher", "Vesper_Sparrow", "Warbling_Vireo", "Western_Grebe", "Western_Gull", "Western_Meadowlark", "Western_Wood_Pewee", "Whip_poor_Will", "White_Pelican", "White_breasted_Kingfisher", "White_breasted_Nuthatch", "White_crowned_Sparrow", "White_eyed_Vireo", "White_necked_Raven", "White_throated_Sparrow", "Wilson_Warbler", "Winter_Wren", "Worm_eating_Warbler", "Yellow_Warbler", "Yellow_bellied_Flycatcher", "Yellow_billed_Cuckoo", "Yellow_breasted_Chat", "Yellow_headed_Blackbird", "Yellow_throated_Vireo" ]
        |> List.map (String.replace "_" " ")
        |> Array.fromList
