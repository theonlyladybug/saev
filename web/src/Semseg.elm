-- https://css-tricks.com/almanac/properties/i/image-rendering/


module Semseg exposing (..)

import Array
import Browser
import Browser.Navigation
import Dict
import File
import Gradio
import Html
import Html.Attributes
import Html.Events
import Http
import Json.Decode as D
import Json.Encode as E
import Parser exposing ((|.), (|=))
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
    | ResetSelectedPatches
    | SetSlider Int String
      -- API responses
    | GotInputExample Requests.Id (Result Gradio.Error Example)
    | GotSaeLatents Requests.Id (Result Gradio.Error (List SaeLatent))
    | GotOrigPreds Requests.Id (Result Gradio.Error Example)
    | GotModPreds Requests.Id (Result Gradio.Error Example)



-- | GotEventId (String -> Cmd Msg) (Result Http.Error String)
-- | GotEventData (String -> Msg) (Result Http.Error String)
-- | ParsedSaeExamples (Result D.Error (List SaeExampleResult))
-- | ParsedImageUrl (Result D.Error (Maybe String))
-- | ParsedTrueLabels (Result D.Error (List SegmentationResult))
-- | ParsedPredLabels (Result D.Error (List SegmentationResult))
-- | ParsedModifiedLabels (Result D.Error (List SegmentationResult))
-- | GetRandomExample
-- | SetExampleInput String


type ImageUploaderMsg
    = Upload
    | DragEnter
    | DragLeave
    | GotFile File.File
    | GotPreview String



-- MODEL


type alias Model =
    { key : Browser.Navigation.Key
    , inputExample : Requests.Requested Example
    , hoveredPatchIndex : Maybe Int
    , selectedPatchIndices : Set.Set Int
    , saeLatents : Requests.Requested (List SaeLatent)

    -- Semantic segmenations
    , trueLabels : Requests.Requested Example
    , origPreds : Requests.Requested Example
    , modPreds : Requests.Requested Example

    -- UI
    , sliders : Dict.Dict Int Float

    -- API
    , gradio : Gradio.Config
    , inputExampleReqId : Requests.Id
    , saeLatentsReqId : Requests.Id
    , origPredsReqId : Requests.Id
    , modPredsReqId : Requests.Id
    }


type alias VitKey =
    String


type alias SaeLatent =
    { latent : Int
    , examples : List HighlightedExample
    }


type alias Example =
    { image : Gradio.Base64Image
    , labels : Gradio.Base64Image
    , index : Int
    }


type alias HighlightedExample =
    { original : Gradio.Base64Image
    , highlighted : Gradio.Base64Image
    , labels : Gradio.Base64Image
    , index : Int
    }


init : () -> Url.Url -> Browser.Navigation.Key -> ( Model, Cmd Msg )
init _ url key =
    let
        ( index, cmd ) =
            case Maybe.andThen .example (Url.Parser.parse urlParser url) of
                Just example ->
                    ( example
                    , Cmd.batch
                        [ getInputExample model.gradio
                            model.inputExampleReqId
                            example
                        , getOrigPreds model.gradio
                            model.origPredsReqId
                            example
                        ]
                    )

                Nothing ->
                    ( 0, Random.generate SetUrl randomExample )

        model =
            { key = key

            -- Image to segment.
            , inputExample = Requests.Initial
            , hoveredPatchIndex = Nothing
            , selectedPatchIndices = Set.empty
            , saeLatents = Requests.Initial
            , trueLabels = Requests.Initial
            , origPreds = Requests.Initial
            , modPreds = Requests.Initial

            -- UI
            , sliders = Dict.empty

            -- API
            , gradio =
                { host = "http://127.0.0.1:7860" }
            , inputExampleReqId = Requests.init
            , saeLatentsReqId = Requests.init
            , origPredsReqId = Requests.init
            , modPredsReqId = Requests.init
            }
    in
    ( model, cmd )



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
                    Debug.log "example" (modBy nImages i)

                inputExampleReqId =
                    Requests.next model.inputExampleReqId

                origPredsReqId =
                    Requests.next model.origPredsReqId
            in
            ( { model
                | inputExample = Requests.Loading
                , selectedPatchIndices = Set.empty
                , saeLatents = Requests.Initial
                , trueLabels = Requests.Loading
                , origPreds = Requests.Loading
                , modPreds = Requests.Initial
                , inputExampleReqId = inputExampleReqId
                , origPredsReqId = origPredsReqId
              }
            , Cmd.batch
                [ getInputExample model.gradio inputExampleReqId example
                , getOrigPreds model.gradio origPredsReqId example
                ]
            )

        GetRandomExample ->
            ( { model
                | inputExample = Requests.Loading
                , selectedPatchIndices = Set.empty
                , saeLatents = Requests.Initial
                , trueLabels = Requests.Loading
                , origPreds = Requests.Loading
                , modPreds = Requests.Initial
              }
            , Random.generate SetUrl randomExample
            )

        HoverPatch i ->
            ( { model | hoveredPatchIndex = Just i }, Cmd.none )

        ResetHoveredPatch ->
            ( { model | hoveredPatchIndex = Nothing }, Cmd.none )

        ToggleSelectedPatch i ->
            let
                patchIndices =
                    if Set.member i model.selectedPatchIndices then
                        Set.remove i model.selectedPatchIndices

                    else
                        Set.insert i model.selectedPatchIndices

                saeLatentsReqId =
                    Requests.next model.saeLatentsReqId

                cmd =
                    case model.inputExample of
                        Requests.Loaded { index } ->
                            getSaeLatents model.gradio
                                saeLatentsReqId
                                index
                                patchIndices

                        _ ->
                            Cmd.none

                saeLatents =
                    if Set.isEmpty patchIndices then
                        Requests.Initial

                    else
                        Requests.Loading
            in
            ( { model
                | selectedPatchIndices = patchIndices
                , saeLatents = saeLatents
                , modPreds = Requests.Initial
                , saeLatentsReqId = saeLatentsReqId
              }
            , cmd
            )

        ResetSelectedPatches ->
            ( { model
                | selectedPatchIndices = Set.empty
                , saeLatents = Requests.Initial
                , modPreds = Requests.Initial
              }
            , Cmd.none
            )

        SetSlider i str ->
            case String.toFloat str of
                Just f ->
                    let
                        sliders =
                            Dict.insert i f model.sliders

                        modPredsReqId =
                            Requests.next model.modPredsReqId

                        cmd =
                            case model.inputExample of
                                Requests.Loaded { index } ->
                                    getModPreds model.gradio
                                        modPredsReqId
                                        index
                                        sliders

                                _ ->
                                    Cmd.none
                    in
                    ( { model | sliders = sliders }
                    , cmd
                    )

                Nothing ->
                    ( model, Cmd.none )

        GotInputExample id result ->
            if Requests.isStale id model.inputExampleReqId then
                ( model, Cmd.none )

            else
                case result of
                    Ok example ->
                        ( { model | inputExample = Requests.Loaded example }
                        , Cmd.none
                        )

                    Err err ->
                        ( { model | inputExample = Requests.Failed (explainGradioError err) }
                        , Cmd.none
                        )

        GotSaeLatents id result ->
            if Requests.isStale id model.saeLatentsReqId then
                ( model, Cmd.none )

            else
                case result of
                    Ok latents ->
                        let
                            sliders =
                                latents
                                    |> List.map .latent
                                    |> List.map (\latent -> ( latent, 0.0 ))
                                    |> Dict.fromList
                        in
                        ( { model | saeLatents = Requests.Loaded latents, sliders = sliders }, Cmd.none )

                    Err err ->
                        ( { model | saeLatents = Requests.Failed (explainGradioError err) }, Cmd.none )

        GotOrigPreds id result ->
            if Requests.isStale id model.origPredsReqId then
                ( model, Cmd.none )

            else
                case result of
                    Ok preds ->
                        ( { model | origPreds = Requests.Loaded preds }, Cmd.none )

                    Err err ->
                        ( { model
                            | origPreds = Requests.Failed (explainGradioError err)
                          }
                        , Cmd.none
                        )

        GotModPreds id result ->
            if Requests.isStale id model.modPredsReqId then
                ( model, Cmd.none )

            else
                case result of
                    Ok preds ->
                        ( { model | modPreds = Requests.Loaded preds }, Cmd.none )

                    Err err ->
                        ( { model
                            | modPreds = Requests.Failed (explainGradioError err)
                          }
                        , Cmd.none
                        )



-- GotEventId fn result ->
--     case result of
--         Ok id ->
--             ( model, fn id )
--         Err err ->
--             let
--                 errMsg =
--                     httpErrToString err "get event id"
--             in
--             ( { model | err = Just errMsg }, Cmd.none )
-- GotEventData decoderFn result ->
--     case result of
--         Ok raw ->
--             case Parser.run eventDataParser raw of
--                 Ok json ->
--                     update (decoderFn json) model
--                 Err _ ->
--                     ( model, Cmd.none )
--         Err err ->
--             let
--                 errMsg =
--                     httpErrToString err "get event result"
--             in
--             ( { model | err = Just errMsg }, Cmd.none )
-- ParsedSaeExamples result ->
--     case result of
--         Ok results ->
--             let
--                 latents =
--                     parseSaeExampleResults results
--                 sliders =
--                     latents
--                         |> List.map (\latent -> ( latent.latent, 0.0 ))
--                         |> Dict.fromList
--             in
--             ( { model | saeLatents = latents, sliders = sliders }, Cmd.none )
--         Err err ->
--             ( { model | err = Just (D.errorToString err) }, Cmd.none )
-- ParsedTrueLabels result ->
--     case result of
--         Ok results ->
--             let
--                 url =
--                     results |> List.filterMap segmentationResultToUrl |> List.head
--             in
--             ( { model | trueLabelsUrl = url }, Cmd.none )
--         Err err ->
--             ( { model | trueLabelsUrl = Nothing, err = Just (D.errorToString err) }
--             , Cmd.none
--             )
-- ParsedPredLabels result ->
--     case result of
--         Ok results ->
--             let
--                 url =
--                     results
--                         |> List.filterMap segmentationResultToUrl
--                         |> List.head
--                 labels =
--                     results
--                         |> List.filterMap segmentationResultToLabels
--                         |> List.head
--                         |> Maybe.withDefault []
--             in
--             ( { model | predLabelsUrl = url, predLabels = labels }, Cmd.none )
--         Err err ->
--             ( { model | predLabelsUrl = Nothing, err = Just (D.errorToString err) }
--             , Cmd.none
--             )
-- ParsedModifiedLabels result ->
--     case result of
--         Ok results ->
--             let
--                 url =
--                     results
--                         |> List.filterMap segmentationResultToUrl
--                         |> List.head
--                 labels =
--                     results
--                         |> List.filterMap segmentationResultToLabels
--                         |> List.head
--                         |> Maybe.withDefault []
--             in
--             ( { model | modifiedLabelsUrl = url, modifiedLabels = labels }, Cmd.none )
--         Err err ->
--             ( { model | modifiedLabelsUrl = Nothing, err = Just (D.errorToString err) }, Cmd.none )
-- ParsedImageUrl result ->
--     case result of
--         Ok maybeUrl ->
--             ( { model | imageUrl = maybeUrl }, Cmd.none )
--         Err err ->
--             ( { model | imageUrl = Nothing, err = Just (D.errorToString err) }, Cmd.none )
-- SetExampleInput str ->
--     case String.toInt str of
--         Just i ->
--             ( { model | exampleIndex = i }
--             , Cmd.batch
--                 [ getImageUrl i
--                 -- , getTrueLabels i
--                 , getPredLabels i
--                 ]
--             )
--         Nothing ->
--             ( model, Cmd.none )
-- GetRandomExample ->
--     ( { model
--         | imageUrl = Nothing
--         , selectedPatchIndices = Set.empty
--         , saeLatents = []
--       }
--     , Random.generate SetExample randomExample
--     )
-- SetSlider i str ->
--     case String.toFloat str of
--         Just f ->
--             let
--                 sliders =
--                     Dict.insert i f model.sliders
--             in
--             ( { model | sliders = sliders }
--             , getModifiedLabels model.exampleIndex sliders
--             )
--         Nothing ->
--             ( model, Cmd.none )


onUrlRequest : Browser.UrlRequest -> Msg
onUrlRequest request =
    NoOp


onUrlChange : Url.Url -> Msg
onUrlChange url =
    Url.Parser.parse urlParser url
        |> Maybe.andThen .example
        |> Maybe.withDefault 0
        |> SetExample


type alias QueryParams =
    { example : Maybe Int
    }


urlParser : Url.Parser.Parser (QueryParams -> a) a
urlParser =
    -- Need to change this when I deploy it.
    Url.Parser.s "web"
        </> Url.Parser.s "apps"
        </> Url.Parser.s "semseg"
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


encodeArgs : List E.Value -> E.Value
encodeArgs args =
    E.object [ ( "data", E.list identity args ) ]


getInputExample : Gradio.Config -> Requests.Id -> Int -> Cmd Msg
getInputExample cfg id img =
    Gradio.get cfg
        "get-img"
        [ E.int img ]
        (Gradio.decodeOne exampleDecoder)
        (GotInputExample id)



-- getImageUrlResult : String -> Cmd Msg
-- getImageUrlResult id =
--     Http.get
--         { url = "http://127.0.0.1:7860/gradio_api/call/get-image/" ++ id
--         , expect =
--             Http.expectString
--                 (GotEventData
--                     (D.decodeString (D.list imgUrlDecoder)
--                         >> Result.map List.head
--                         >> ParsedImageUrl
--                     )
--                 )
--         }
-- getTrueLabels : Int -> Cmd Msg
-- getTrueLabels img =
--     Http.post
--         { url = "http://127.0.0.1:7860/gradio_api/call/get-true-labels"
--         , body =
--             Http.jsonBody (encodeArgs [ E.int img ])
--         , expect = Http.expectJson (GotEventId getTrueLabelsResult) eventIdDecoder
--         }
-- getTrueLabelsResult : String -> Cmd Msg
-- getTrueLabelsResult id =
--     Http.get
--         { url = "http://127.0.0.1:7860/gradio_api/call/get-true-labels/" ++ id
--         , expect =
--             Http.expectString
--                 (GotEventData
--                     (D.decodeString (D.list segmentationResultDecoder)
--                         >> ParsedTrueLabels
--                     )
--                 )
--         }


getOrigPreds : Gradio.Config -> Requests.Id -> Int -> Cmd Msg
getOrigPreds cfg id img =
    Gradio.get cfg
        "get-orig-preds"
        [ E.int img ]
        (Gradio.decodeOne exampleDecoder)
        (GotOrigPreds id)


getModPreds : Gradio.Config -> Requests.Id -> Int -> Dict.Dict Int Float -> Cmd Msg
getModPreds cfg id img sliders =
    Gradio.get cfg
        "get-mod-preds"
        [ E.int img
        , E.dict String.fromInt E.float sliders
        ]
        (Gradio.decodeOne exampleDecoder)
        (GotModPreds id)



-- getModifiedLabels : Int -> Dict.Dict Int Float -> Cmd Msg
-- getModifiedLabels img sliders =
--     let
--         ( latents, values ) =
--             sliders
--                 |> Dict.toList
--                 |> List.unzip
--                 |> Tuple.mapBoth Array.fromList Array.fromList
--     in
--     Http.post
--         { url = "http://127.0.0.1:7860/gradio_api/call/get-modified-labels"
--         , body =
--             Http.jsonBody
--                 (encodeArgs
--                     [ E.int img
--                     , E.int (Array.get 0 latents |> Maybe.withDefault -1)
--                     , E.int (Array.get 1 latents |> Maybe.withDefault -1)
--                     , E.int (Array.get 2 latents |> Maybe.withDefault -1)
--                     , E.float (Array.get 0 values |> Maybe.withDefault 0.0)
--                     , E.float (Array.get 1 values |> Maybe.withDefault 0.0)
--                     , E.float (Array.get 2 values |> Maybe.withDefault 0.0)
--                     ]
--                 )
--         , expect = Http.expectJson (GotEventId getModifiedLabelsResult) eventIdDecoder
--         }
-- getModifiedLabelsResult : String -> Cmd Msg
-- getModifiedLabelsResult id =
--     Http.get
--         { url = "http://127.0.0.1:7860/gradio_api/call/get-modified-labels/" ++ id
--         , expect =
--             Http.expectString
--                 (GotEventData
--                     (D.decodeString (D.list segmentationResultDecoder)
--                         >> ParsedModifiedLabels
--                     )
--                 )
--         }


getSaeLatents : Gradio.Config -> Requests.Id -> Int -> Set.Set Int -> Cmd Msg
getSaeLatents cfg id img patches =
    Gradio.get cfg
        "get-sae-latents"
        [ E.int img
        , Set.toList patches |> E.list E.int
        ]
        (Gradio.decodeOne
            (D.list
                (D.map2 SaeLatent
                    (D.field "latent" D.int)
                    (D.field "examples" (D.list highlightedExampleDecoder))
                )
            )
        )
        (GotSaeLatents id)



-- saeExampleOutputToLatent : SaeExampleResult -> Maybe Int
-- saeExampleOutputToLatent result =
--     case result of
--         SaeExampleLatent latent ->
--             Just latent
--         _ ->
--             Nothing
-- saeExampleResultDecoder : D.Decoder SaeExampleResult
-- saeExampleResultDecoder =
--     D.oneOf
--         [ imgUrlDecoder |> D.map SaeExampleUrl
--         , D.int |> D.map SaeExampleLatent
--         , D.null SaeExampleMissing
--         ]
-- parseSaeExampleResults : List SaeExampleResult -> List SaeLatent
-- parseSaeExampleResults results =
--     let
--         ( _, _, parsed ) =
--             parseSaeExampleResultsHelper results [] []
--     in
--     parsed
-- parseSaeExampleResultsHelper : List SaeExampleResult -> List (Maybe String) -> List SaeLatent -> ( List SaeExampleResult, List (Maybe String), List SaeLatent )
-- parseSaeExampleResultsHelper unparsed urls parsed =
--     case unparsed of
--         [] ->
--             ( [], urls, parsed )
--         (SaeExampleUrl url) :: rest ->
--             parseSaeExampleResultsHelper
--                 rest
--                 (urls ++ [ Just url ])
--                 parsed
--         (SaeExampleLatent latent) :: rest ->
--             parseSaeExampleResultsHelper
--                 rest
--                 (List.drop nSaeExamplesPerLatent urls)
--                 (parsed
--                     ++ [ { latent = latent
--                          , urls =
--                             urls
--                                 |> List.take 4
--                                 |> List.filterMap identity
--                          }
--                        ]
--                 )
--         SaeExampleMissing :: rest ->
--             parseSaeExampleResultsHelper
--                 rest
--                 (urls ++ [ Nothing ])
--                 parsed


type SegmentationResult
    = SegmentationUrl String
    | SegmentationLabels (List Int)


segmentationResultToUrl : SegmentationResult -> Maybe String
segmentationResultToUrl result =
    case result of
        SegmentationUrl url ->
            Just url

        _ ->
            Nothing


segmentationResultToLabels : SegmentationResult -> Maybe (List Int)
segmentationResultToLabels result =
    case result of
        SegmentationLabels labels ->
            Just labels

        _ ->
            Nothing


exampleDecoder : D.Decoder Example
exampleDecoder =
    D.map3 Example
        (D.field "orig_url" Gradio.base64ImageDecoder)
        (D.field "seg_url" Gradio.base64ImageDecoder)
        (D.field "index" D.int)


highlightedExampleDecoder : D.Decoder HighlightedExample
highlightedExampleDecoder =
    D.map4 HighlightedExample
        (D.field "orig_url" Gradio.base64ImageDecoder)
        (D.field "highlighted_url" Gradio.base64ImageDecoder)
        (D.field "seg_url" Gradio.base64ImageDecoder)
        (D.field "index" D.int)


imgUrlDecoder : D.Decoder String
imgUrlDecoder =
    D.field "url" D.string
        |> D.map (String.replace "gradio_api/gradio_api" "gradio_api")
        |> D.map (String.replace "gra/gradio_api" "gradio_api")
        |> D.map (String.replace "gradio_ap/gradio_api" "gradio_api")
        |> D.map (String.replace "gradio_api/ca/gradio_api" "gradio_api")
        |> D.map (String.replace "gradio_api/call/gradio_api" "gradio_api")


view : Model -> Browser.Document Msg
view model =
    { title = "Semantic Segmentation"
    , body =
        [ Html.header [] []
        , Html.main_
            [ Html.Attributes.class "w-full min-h-screen p-1 md:p-2 lg:p-4 bg-gray-50 space-y-4" ]
            [ Html.h2
                []
                [ Html.text "SAEs for Scientifically Rigorous Interpretation of Semantic Segmentation Models" ]
            , viewControls model.inputExample
            , Html.div
                [ Html.Attributes.class "flex flex-row gap-2 items-stretch" ]
                [ viewGriddedImage model
                    (Requests.map .image model.inputExample)
                    "Input Image"
                    "Wait just a second..."
                , viewGriddedImage model
                    (Requests.map .labels model.inputExample)
                    "True Labels"
                    "Wait just a second..."
                , viewGriddedImage model
                    (Requests.map .labels model.origPreds)
                    "Predicted Segmentation"
                    "Wait just a second..."
                , viewGriddedImage model
                    (Requests.map .labels model.modPreds)
                    "Modified Segmentation"
                    "Modify the ViT's representations using the sliders below."
                ]
            , Html.div
                [ Html.Attributes.class "flex flex-row" ]
                [ viewSaeLatents model.selectedPatchIndices model.saeLatents model.sliders

                -- , viewLegend (Set.fromList (model.predLabels ++ model.modifiedLabels))
                ]
            ]
        ]
    }


viewErr : String -> Html.Html Msg
viewErr err =
    Html.div
        [ Html.Attributes.class "relative rounded-lg border border-red-200 bg-red-50 p-4 m-4" ]
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


viewControls : Requests.Requested Example -> Html.Html Msg
viewControls requestedExample =
    case requestedExample of
        Requests.Initial ->
            Html.div
                [ Html.Attributes.class "flex flex-row gap-2" ]
                [ viewButtonDisabled "Previous"
                , viewButton GetRandomExample "Random"
                , viewButtonDisabled "Next"
                ]

        Requests.Loading ->
            Html.div
                [ Html.Attributes.class "flex flex-row gap-2" ]
                [ viewButtonDisabled "Previous"
                , viewButton GetRandomExample "Random"
                , viewButtonDisabled "Next"
                ]

        Requests.Loaded example ->
            Html.div
                [ Html.Attributes.class "flex flex-row gap-2" ]
                [ viewButton (SetUrl (example.index - 1)) "Previous"
                , viewButton GetRandomExample "Random"
                , viewButton (SetUrl (example.index + 1)) "Next"
                ]

        Requests.Failed err ->
            Html.div
                [ Html.Attributes.class "flex flex-row gap-2" ]
                [ viewButtonDisabled "Previous"
                , viewButton GetRandomExample "Random"
                , viewButtonDisabled "Next"
                ]


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


viewButtonDisabled : String -> Html.Html Msg
viewButtonDisabled title =
    Html.button
        [ Html.Attributes.disabled True
        , Html.Attributes.class "flex-1 rounded-lg px-2 py-1 transition-colors"
        , Html.Attributes.class "border border-sky-300 hover:border-sky-400"
        , Html.Attributes.class "bg-sky-100 hover:bg-sky-200"
        , Html.Attributes.class "text-gray-700 hover:text-gray-900"
        , Html.Attributes.class "focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2"
        , Html.Attributes.class "active:bg-gray-300"
        ]
        [ Html.text title ]


viewGriddedImage : Model -> Requests.Requested Gradio.Base64Image -> String -> String -> Html.Html Msg
viewGriddedImage model reqImage title callToAction =
    case reqImage of
        Requests.Initial ->
            Html.div
                []
                [ Html.p
                    [ Html.Attributes.class "italic" ]
                    [ Html.text callToAction ]
                ]

        Requests.Loading ->
            Html.div
                []
                [ Html.p
                    [ Html.Attributes.class "italic" ]
                    [ Html.text "Loading..." ]
                ]

        Requests.Failed err ->
            viewErr err

        Requests.Loaded image ->
            Html.div []
                [ Html.p
                    [ Html.Attributes.class "text-center" ]
                    [ Html.text title ]
                , Html.div
                    [ Html.Attributes.class "relative inline-block" ]
                    [ Html.div
                        [ Html.Attributes.class "absolute grid grid-rows-[repeat(16,_14px)] grid-cols-[repeat(16,_14px)] md:grid-rows-[repeat(16,_21px)] md:grid-cols-[repeat(16,_21px)]" ]
                        (List.map
                            (viewGridCell model.hoveredPatchIndex model.selectedPatchIndices)
                            (List.range 0 255)
                        )
                    , Html.img
                        [ Html.Attributes.class "block w-[224px] h-[224px] md:w-[336px] md:h-[336px]"
                        , Html.Attributes.src (Gradio.base64ImageToString image)
                        , Html.Attributes.style "image-rendering" "pixelated"
                        ]
                        []
                    ]
                ]


viewGridCell : Maybe Int -> Set.Set Int -> Int -> Html.Html Msg
viewGridCell hovered selected self =
    let
        classes =
            (case hovered of
                Just h ->
                    if h == self then
                        [ "border-2 border-rose-600 border-dashed" ]

                    else
                        []

                Nothing ->
                    []
            )
                ++ (if Set.member self selected then
                        [ "bg-rose-600/50" ]

                    else
                        []
                   )
    in
    Html.div
        ([ Html.Attributes.class "w-[14px] h-[14px] md:w-[21px] md:h-[21px]"
         , Html.Events.onMouseEnter (HoverPatch self)
         , Html.Events.onMouseLeave ResetHoveredPatch
         , Html.Events.onClick (ToggleSelectedPatch self)
         ]
            ++ List.map Html.Attributes.class classes
        )
        []



-- viewLegend : Set.Set Int -> Html.Html Msg
-- viewLegend classes =
--     Html.div
--         [ Html.Attributes.id "legend" ]
--         [ Html.p [] [ Html.text "Legend" ]
--         , Html.div
--             []
--             (Set.toList classes
--                 |> List.sort
--                 |> List.filter (\x -> x > 0)
--                 |> List.map viewClassIcon
--             )
--         ]
-- viewClassIcon : Int -> Html.Html Msg
-- viewClassIcon class =
--     let
--         color =
--             case Array.get (class - 1) colors of
--                 Just ( r, g, b ) ->
--                     "rgb(" ++ String.fromInt r ++ " " ++ String.fromInt g ++ " " ++ String.fromInt b ++ ")"
--                 Nothing ->
--                     "red"
--         classname =
--             case Array.get (class - 1) classnames of
--                 Just names ->
--                     names |> List.take 2 |> String.join ", "
--                 Nothing ->
--                     "no classname found"
--     in
--     Html.div
--         [ Html.Attributes.class "flex flex-row gap-1 items-center" ]
--         [ Html.span
--             [ Html.Attributes.class "w-4 h-4"
--             , Html.Attributes.style "background-color" color
--             ]
--             []
--         , Html.span
--             []
--             [ Html.text (classname ++ " (class " ++ String.fromInt class ++ ")") ]
--         ]
-- viewImageSeg : Maybe String -> String -> Html.Html Msg
-- viewImageSeg maybeUrl title =
--     case maybeUrl of
--         Just url ->
--             Html.div
--                 []
--                 [ Html.img
--                     [ Html.Attributes.src url
--                     , Html.Attributes.style "width" "448px"
--                     ]
--                     []
--                 , Html.p [] [ Html.text title ]
--                 ]
--         Nothing ->
--             Html.div [] [ Html.text ("Loading '" ++ title ++ "'...") ]


viewSaeLatents : Set.Set Int -> Requests.Requested (List SaeLatent) -> Dict.Dict Int Float -> Html.Html Msg
viewSaeLatents selected requestedLatents values =
    case requestedLatents of
        Requests.Initial ->
            Html.p []
                [ Html.text "Click on the image above to explain model predictions." ]

        Requests.Loading ->
            Html.p []
                [ Html.text "Loading similar patches..." ]

        Requests.Failed err ->
            viewErr err

        Requests.Loaded latents ->
            Html.div []
                [ Html.p []
                    [ Html.span [ Html.Attributes.class "bg-rose-600 p-1 rounded" ] [ Html.text "These patches" ]
                    , Html.text " above are like "
                    , Html.span [ Html.Attributes.class "bg-rose-600 p-1 rounded" ] [ Html.text "these patches" ]
                    , Html.text " below. (Not what you expected? Add more patches and get a larger "
                    , Html.a [ Html.Attributes.href "https://simple.wikipedia.org/wiki/Sampling_(statistics)", Html.Attributes.class "text-blue-500 underline" ] [ Html.text "sample size" ]
                    , Html.text ")"
                    ]
                , Html.div []
                    (List.filterMap
                        (\latent -> Maybe.map (\f -> viewSaeLatent latent f) (Dict.get latent.latent values))
                        latents
                    )
                ]


viewSaeLatent : SaeLatent -> Float -> Html.Html Msg
viewSaeLatent latent value =
    Html.div
        [ Html.Attributes.class "flex flex-row gap-2 mt-2" ]
        (List.map viewHighlightedExample latent.examples
            ++ [ Html.div
                    [ Html.Attributes.class "flex flex-col gap-2" ]
                    [ Html.input
                        [ Html.Attributes.type_ "range"
                        , Html.Attributes.min "-10"
                        , Html.Attributes.max "10"
                        , Html.Attributes.value (String.fromFloat value)
                        , Html.Events.onInput (SetSlider latent.latent)
                        ]
                        []
                    , Html.p
                        []
                        [ Html.text ("Latent 24K/" ++ String.fromInt latent.latent) ]
                    , Html.p
                        []
                        [ Html.text ("Value:" ++ String.fromFloat value) ]
                    ]
               ]
        )


viewHighlightedExample : HighlightedExample -> Html.Html Msg
viewHighlightedExample { original, highlighted } =
    Html.img
        [ Html.Attributes.src (Gradio.base64ImageToString highlighted)
        , Html.Attributes.class "max-w-36 h-auto"
        ]
        []



-- CONSTANTS


nImages =
    2000


nSaeLatents =
    3


nSaeExamplesPerLatent =
    4


randomExample : Random.Generator Int
randomExample =
    Random.int 0 (nImages - 1)



-- colors : Array.Array ( Int, Int, Int )
-- colors =
--     [ ( 51, 0, 0 ), ( 204, 0, 102 ), ( 0, 255, 0 ), ( 102, 51, 51 ), ( 153, 204, 51 ), ( 51, 51, 153 ), ( 102, 0, 51 ), ( 153, 153, 0 ), ( 51, 102, 204 ), ( 204, 255, 0 ), ( 204, 102, 0 ), ( 204, 255, 153 ), ( 102, 102, 255 ), ( 255, 204, 255 ), ( 51, 255, 0 ), ( 0, 102, 51 ), ( 102, 102, 0 ), ( 0, 0, 255 ), ( 255, 153, 204 ), ( 204, 204, 0 ), ( 0, 153, 153 ), ( 153, 102, 204 ), ( 255, 204, 0 ), ( 204, 204, 153 ), ( 255, 51, 0 ), ( 51, 51, 0 ), ( 153, 51, 51 ), ( 0, 0, 102 ), ( 102, 255, 204 ), ( 204, 51, 255 ), ( 255, 204, 204 ), ( 0, 0, 153 ), ( 0, 102, 153 ), ( 153, 0, 51 ), ( 51, 51, 102 ), ( 255, 153, 0 ), ( 204, 153, 0 ), ( 153, 102, 153 ), ( 51, 204, 204 ), ( 51, 51, 255 ), ( 153, 204, 102 ), ( 102, 204, 153 ), ( 153, 153, 204 ), ( 0, 51, 204 ), ( 204, 204, 102 ), ( 0, 51, 153 ), ( 0, 102, 0 ), ( 51, 0, 102 ), ( 153, 255, 0 ), ( 153, 255, 102 ), ( 102, 102, 51 ), ( 153, 0, 255 ), ( 204, 255, 102 ), ( 102, 0, 255 ), ( 255, 204, 153 ), ( 102, 51, 0 ), ( 102, 204, 102 ), ( 0, 102, 204 ), ( 51, 204, 0 ), ( 255, 102, 102 ), ( 153, 255, 204 ), ( 51, 204, 51 ), ( 0, 0, 0 ), ( 255, 0, 255 ), ( 153, 0, 153 ), ( 255, 204, 51 ), ( 51, 0, 51 ), ( 102, 204, 255 ), ( 153, 204, 153 ), ( 153, 102, 0 ), ( 102, 204, 204 ), ( 204, 204, 204 ), ( 255, 0, 0 ), ( 255, 255, 51 ), ( 0, 255, 102 ), ( 204, 153, 102 ), ( 204, 153, 153 ), ( 102, 51, 153 ), ( 51, 102, 0 ), ( 204, 51, 153 ), ( 153, 51, 255 ), ( 102, 0, 204 ), ( 204, 102, 153 ), ( 204, 0, 204 ), ( 102, 51, 102 ), ( 0, 153, 51 ), ( 153, 153, 51 ), ( 255, 102, 0 ), ( 255, 153, 153 ), ( 153, 0, 102 ), ( 51, 204, 255 ), ( 102, 255, 102 ), ( 255, 255, 204 ), ( 51, 51, 204 ), ( 153, 102, 51 ), ( 153, 153, 255 ), ( 51, 153, 0 ), ( 204, 0, 255 ), ( 102, 255, 0 ), ( 153, 102, 255 ), ( 204, 102, 255 ), ( 204, 0, 0 ), ( 102, 153, 255 ), ( 204, 102, 204 ), ( 204, 51, 102 ), ( 0, 255, 153 ), ( 153, 204, 204 ), ( 255, 0, 102 ), ( 102, 51, 204 ), ( 255, 51, 204 ), ( 51, 204, 153 ), ( 153, 153, 102 ), ( 153, 204, 0 ), ( 153, 102, 102 ), ( 204, 153, 255 ), ( 153, 0, 204 ), ( 102, 0, 0 ), ( 255, 51, 255 ), ( 0, 204, 153 ), ( 255, 153, 51 ), ( 0, 255, 204 ), ( 51, 102, 153 ), ( 255, 51, 51 ), ( 102, 255, 51 ), ( 0, 0, 204 ), ( 102, 255, 153 ), ( 0, 204, 255 ), ( 0, 102, 102 ), ( 102, 51, 255 ), ( 255, 0, 204 ), ( 51, 255, 153 ), ( 204, 0, 51 ), ( 153, 51, 204 ), ( 204, 102, 51 ), ( 255, 255, 0 ), ( 51, 51, 51 ), ( 0, 153, 0 ), ( 51, 255, 102 ), ( 51, 102, 255 ), ( 102, 153, 0 ), ( 102, 153, 204 ), ( 51, 0, 255 ), ( 102, 153, 153 ), ( 153, 51, 102 ), ( 204, 255, 51 ), ( 204, 204, 51 ), ( 0, 204, 51 ), ( 255, 102, 153 ), ( 204, 102, 102 ), ( 102, 0, 102 ), ( 51, 153, 204 ), ( 255, 255, 255 ), ( 0, 102, 255 ), ( 51, 102, 51 ), ( 204, 0, 153 ), ( 102, 153, 102 ), ( 102, 0, 153 ), ( 153, 255, 153 ), ( 0, 153, 102 ), ( 102, 204, 0 ), ( 0, 255, 51 ), ( 153, 204, 255 ), ( 153, 51, 153 ), ( 0, 51, 255 ), ( 51, 255, 51 ), ( 255, 102, 51 ), ( 102, 102, 204 ), ( 102, 153, 51 ), ( 0, 204, 0 ), ( 102, 204, 51 ), ( 255, 102, 255 ), ( 255, 204, 102 ), ( 102, 102, 102 ), ( 255, 102, 204 ), ( 51, 0, 153 ), ( 255, 0, 51 ), ( 102, 102, 153 ), ( 255, 153, 102 ), ( 204, 255, 204 ), ( 51, 0, 204 ), ( 0, 0, 51 ), ( 51, 255, 255 ), ( 204, 51, 0 ), ( 153, 51, 0 ), ( 51, 153, 102 ), ( 102, 255, 255 ), ( 255, 153, 255 ), ( 204, 255, 255 ), ( 204, 153, 204 ), ( 255, 0, 153 ), ( 51, 102, 102 ), ( 153, 255, 255 ), ( 255, 255, 153 ), ( 204, 51, 204 ), ( 153, 153, 153 ), ( 51, 153, 255 ), ( 51, 153, 51 ), ( 0, 153, 255 ), ( 0, 51, 51 ), ( 0, 51, 102 ), ( 153, 0, 0 ), ( 204, 51, 51 ), ( 0, 153, 204 ), ( 153, 255, 51 ), ( 255, 255, 102 ), ( 204, 204, 255 ), ( 0, 204, 102 ), ( 255, 51, 102 ), ( 0, 255, 255 ), ( 51, 153, 153 ), ( 51, 204, 102 ), ( 51, 255, 204 ), ( 255, 51, 153 ), ( 0, 51, 0 ), ( 0, 204, 204 ), ( 204, 153, 51 ) ]
--         |> Array.fromList
--         -- Fixed colors for example 3122
--         |> Array.set 2 ( 201, 249, 255 )
--         |> Array.set 4 ( 151, 204, 4 )
--         |> Array.set 16 ( 54, 48, 32 )
--         |> Array.set 26 ( 45, 125, 210 )
--         |> Array.set 46 ( 238, 185, 2 )
--         |> Array.set 72 ( 76, 46, 5 )
--         |> Array.set 94 ( 12, 15, 10 )
-- classnames : Array.Array (List String)
-- classnames =
--     Array.fromList
--         [ [ "wall" ]
--         , [ "building", "edifice" ]
--         , [ "sky" ]
--         , [ "floor", "flooring" ]
--         , [ "tree" ]
--         , [ "ceiling" ]
--         , [ "road", "route" ]
--         , [ "bed" ]
--         , [ "windowpane", "window" ]
--         , [ "grass" ]
--         , [ "cabinet" ]
--         , [ "sidewalk", "pavement" ]
--         , [ "person", "individual" ]
--         , [ "earth", "ground" ]
--         , [ "door", "double door" ]
--         , [ "table" ]
--         , [ "mountain" ]
--         , [ "plant", "flora" ]
--         , [ "curtain", "drape" ]
--         , [ "chair" ]
--         , [ "car", "auto" ]
--         , [ "water" ]
--         , [ "painting", "picture" ]
--         , [ "sofa", "couch" ]
--         , [ "shelf" ]
--         , [ "house" ]
--         , [ "sea" ]
--         , [ "mirror" ]
--         , [ "rug", "carpet" ]
--         , [ "field" ]
--         , [ "armchair" ]
--         , [ "seat" ]
--         , [ "fence", "fencing" ]
--         , [ "desk" ]
--         , [ "rock", "stone" ]
--         , [ "wardrobe", "closet" ]
--         , [ "lamp" ]
--         , [ "bathtub", "bathing tub" ]
--         , [ "railing", "rail" ]
--         , [ "cushion" ]
--         , [ "base", "pedestal" ]
--         , [ "box" ]
--         , [ "column", "pillar" ]
--         , [ "signboard", "sign" ]
--         , [ "chest of drawers", "chest" ]
--         , [ "counter" ]
--         , [ "sand" ]
--         , [ "sink" ]
--         , [ "skyscraper" ]
--         , [ "fireplace", "hearth" ]
--         , [ "refrigerator", "icebox" ]
--         , [ "grandstand", "covered stand" ]
--         , [ "path" ]
--         , [ "stairs", "steps" ]
--         , [ "runway" ]
--         , [ "case", "display case" ]
--         , [ "pool table", "billiard table" ]
--         , [ "pillow" ]
--         , [ "screen door", "screen" ]
--         , [ "stairway", "staircase" ]
--         , [ "river" ]
--         , [ "bridge", "span" ]
--         , [ "bookcase" ]
--         , [ "blind", "screen" ]
--         , [ "coffee table", "cocktail table" ]
--         , [ "toilet", "can" ]
--         , [ "flower" ]
--         , [ "book" ]
--         , [ "hill" ]
--         , [ "bench" ]
--         , [ "countertop" ]
--         , [ "stove", "kitchen stove" ]
--         , [ "palm", "palm tree" ]
--         , [ "kitchen island" ]
--         , [ "computer", "computing machine" ]
--         , [ "swivel chair" ]
--         , [ "boat" ]
--         , [ "bar" ]
--         , [ "arcade machine" ]
--         , [ "hovel", "hut" ]
--         , [ "bus", "autobus" ]
--         , [ "towel" ]
--         , [ "light", "light source" ]
--         , [ "truck", "motortruck" ]
--         , [ "tower" ]
--         , [ "chandelier", "pendant" ]
--         , [ "awning", "sunshade" ]
--         , [ "streetlight", "street lamp" ]
--         , [ "booth", "cubicle" ]
--         , [ "television receiver", "television" ]
--         , [ "airplane", "aeroplane" ]
--         , [ "dirt track" ]
--         , [ "apparel", "wearing apparel" ]
--         , [ "pole" ]
--         , [ "land", "ground" ]
--         , [ "bannister", "banister" ]
--         , [ "escalator", "moving staircase" ]
--         , [ "ottoman", "pouf" ]
--         , [ "bottle" ]
--         , [ "buffet", "counter" ]
--         , [ "poster", "posting" ]
--         , [ "stage" ]
--         , [ "van" ]
--         , [ "ship" ]
--         , [ "fountain" ]
--         , [ "conveyer belt", "conveyor belt" ]
--         , [ "canopy" ]
--         , [ "washer", "automatic washer" ]
--         , [ "plaything", "toy" ]
--         , [ "swimming pool", "swimming bath" ]
--         , [ "stool" ]
--         , [ "barrel", "cask" ]
--         , [ "basket", "handbasket" ]
--         , [ "waterfall", "falls" ]
--         , [ "tent", "collapsible shelter" ]
--         , [ "bag" ]
--         , [ "minibike", "motorbike" ]
--         , [ "cradle" ]
--         , [ "oven" ]
--         , [ "ball" ]
--         , [ "food", "solid food" ]
--         , [ "step", "stair" ]
--         , [ "tank", "storage tank" ]
--         , [ "trade name", "brand name" ]
--         , [ "microwave", "microwave oven" ]
--         , [ "pot", "flowerpot" ]
--         , [ "animal", "animate being" ]
--         , [ "bicycle", "bike" ]
--         , [ "lake" ]
--         , [ "dishwasher", "dish washer" ]
--         , [ "screen", "silver screen" ]
--         , [ "blanket", "cover" ]
--         , [ "sculpture" ]
--         , [ "hood", "exhaust hood" ]
--         , [ "sconce" ]
--         , [ "vase" ]
--         , [ "traffic light", "traffic signal" ]
--         , [ "tray" ]
--         , [ "ashcan", "trash can" ]
--         , [ "fan" ]
--         , [ "pier", "wharf" ]
--         , [ "crt screen" ]
--         , [ "plate" ]
--         , [ "monitor", "monitoring device" ]
--         , [ "bulletin board", "notice board" ]
--         , [ "shower" ]
--         , [ "radiator" ]
--         , [ "glass", "drinking glass" ]
--         , [ "clock" ]
--         , [ "flag" ]
--         ]
